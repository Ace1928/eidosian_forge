from __future__ import annotations
import concurrent.futures
import dataclasses
import functools
import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import (
from langchain_core._api import warn_deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables import config as runnable_config
from langchain_core.runnables import utils as runnable_utils
from langchain_core.tracers.evaluation import (
from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client
from langsmith.env import get_git_info, get_langchain_env_var_metadata
from langsmith.evaluation import (
from langsmith.evaluation import (
from langsmith.run_helpers import as_runnable, is_traceable_function
from langsmith.schemas import Dataset, DataType, Example, Run, TracerSession
from langsmith.utils import LangSmithError
from requests import HTTPError
from typing_extensions import TypedDict
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.schema import (
from langchain.smith import evaluation as smith_eval
from langchain.smith.evaluation import config as smith_eval_config
from langchain.smith.evaluation import name_generation, progress
def run_on_dataset(client: Optional[Client], dataset_name: str, llm_or_chain_factory: MODEL_OR_CHAIN_FACTORY, *, evaluation: Optional[smith_eval.RunEvalConfig]=None, dataset_version: Optional[Union[datetime, str]]=None, concurrency_level: int=5, project_name: Optional[str]=None, project_metadata: Optional[Dict[str, Any]]=None, verbose: bool=False, revision_id: Optional[str]=None, **kwargs: Any) -> Dict[str, Any]:
    input_mapper = kwargs.pop('input_mapper', None)
    if input_mapper:
        warn_deprecated('0.0.305', message=_INPUT_MAPPER_DEP_WARNING, pending=True)
    tags = kwargs.pop('tags', None)
    if tags:
        warn_deprecated('0.1.9', message='The tags argument is deprecated and will be removed in a future release. Please specify project_metadata instead.', pending=True)
    if revision_id is None:
        revision_id = get_langchain_env_var_metadata().get('revision_id')
    if kwargs:
        warn_deprecated('0.0.305', message=f'The following arguments are deprecated and will be removed in a future release: {kwargs.keys()}.', removal='0.0.305')
    client = client or Client()
    container = _DatasetRunContainer.prepare(client, dataset_name, llm_or_chain_factory, project_name, evaluation, tags, input_mapper, concurrency_level, project_metadata=project_metadata, revision_id=revision_id, dataset_version=dataset_version)
    if concurrency_level == 0:
        batch_results = [_run_llm_or_chain(example, config, llm_or_chain_factory=container.wrapped_model, input_mapper=input_mapper) for example, config in zip(container.examples, container.configs)]
    else:
        with runnable_config.get_executor_for_config(container.configs[0]) as executor:
            batch_results = list(executor.map(functools.partial(_run_llm_or_chain, llm_or_chain_factory=container.wrapped_model, input_mapper=input_mapper), container.examples, container.configs))
    return container.finish(batch_results, verbose=verbose)