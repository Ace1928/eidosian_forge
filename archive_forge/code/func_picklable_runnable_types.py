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
@lru_cache
def picklable_runnable_types():
    """
    Runnable types that can be pickled and unpickled by cloudpickle.
    """
    from langchain.chat_models.base import SimpleChatModel
    from langchain.prompts import ChatPromptTemplate
    types = (SimpleChatModel, ChatPromptTemplate)
    try:
        from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
        types += (RunnableLambda, RunnablePassthrough)
    except ImportError:
        pass
    return types