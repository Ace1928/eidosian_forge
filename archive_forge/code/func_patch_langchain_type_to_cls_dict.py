import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import types
import warnings
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple
import cloudpickle
import yaml
from packaging import version
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.class_utils import _get_class_from_string
@contextlib.contextmanager
def patch_langchain_type_to_cls_dict():
    """Patch LangChain's type_to_cls_dict config to handle unsupported types like ChatOpenAI.

    The type_to_cls_dict is a hard-coded dictionary in LangChain code base that defines the mapping
    between the LLM type e.g. "openai" to the loader function for the corresponding LLM class.
    However, this dictionary doesn't contain some chat models like ChatOpenAI, AzureChatOpenAI,
    which makes it unable to save and load chains with these models. Ideally, the config should
    be updated in the LangChain code base, but similar requests have been rejected multiple times
    in the past, because they consider this serde method to be deprecated, and instead prompt
    users to use their new serde method https://github.com/langchain-ai/langchain/pull/8164#issuecomment-1659723157.
    However, we can't simply migrate to the new method because it doesn't support common chains
    like RetrievalQA, AgentExecutor, etc.
    Therefore, we apply a hacky solution to patch the type_to_cls_dict from our side to support
    these models, until a better solution is provided by LangChain.
    """

    def _load_chat_openai():
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI

    def _load_azure_chat_openai():
        from langchain.chat_models import AzureChatOpenAI
        return AzureChatOpenAI

    def _patched_get_type_to_cls_dict(original):

        def _wrapped():
            return {**original(), 'openai-chat': _load_chat_openai, 'azure-openai-chat': _load_azure_chat_openai}
        return _wrapped
    modules_to_patch = ['langchain.llms', 'langchain_community.llms.loading']
    originals = {}
    for module_name in modules_to_patch:
        try:
            module = importlib.import_module(module_name)
            originals[module_name] = module.get_type_to_cls_dict
        except (ImportError, AttributeError):
            continue
        module.get_type_to_cls_dict = _patched_get_type_to_cls_dict(originals[module_name])
    try:
        yield
    except ValueError as e:
        if (m := _CHAT_MODELS_ERROR_MSG.search(str(e))):
            model_name = 'ChatOpenAI' if m.group(1) == 'openai-chat' else 'AzureChatOpenAI'
            raise mlflow.MlflowException(f'Loading {model_name} chat model is not supported in MLflow with the current version of LangChain. Please upgrade LangChain to 0.0.307 or above by running `pip install langchain>=0.0.307`.') from e
        else:
            raise
    finally:
        for module_name, original_impl in originals.items():
            module = importlib.import_module(module_name)
            module.get_type_to_cls_dict = original_impl