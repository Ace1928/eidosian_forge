from __future__ import annotations
import collections
import datetime
import functools
import importlib
import importlib.metadata
import io
import json
import logging
import os
import random
import re
import socket
import sys
import threading
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue, Queue
from typing import (
from urllib import parse as urllib_parse
import orjson
import requests
from requests import adapters as requests_adapters
from urllib3.util import Retry
import langsmith
from langsmith import env as ls_env
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
def create_example_from_run(self, run: ls_schemas.Run, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, created_at: Optional[datetime.datetime]=None) -> ls_schemas.Example:
    """Add an example (row) to an LLM-type dataset."""
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
        dataset_name = None
    dataset_type = self._get_data_type_cached(dataset_id)
    if dataset_type == ls_schemas.DataType.llm:
        if run.run_type != 'llm':
            raise ValueError(f"Run type {run.run_type} is not supported for dataset of type 'LLM'")
        try:
            prompt = ls_utils.get_prompt_from_inputs(run.inputs)
        except ValueError:
            raise ValueError(f'Error converting LLM run inputs to prompt for run {run.id} with inputs {run.inputs}')
        inputs: Dict[str, Any] = {'input': prompt}
        if not run.outputs:
            outputs: Optional[Dict[str, Any]] = None
        else:
            try:
                generation = ls_utils.get_llm_generation_from_outputs(run.outputs)
            except ValueError:
                raise ValueError(f'Error converting LLM run outputs to generation for run {run.id} with outputs {run.outputs}')
            outputs = {'output': generation}
    elif dataset_type == ls_schemas.DataType.chat:
        if run.run_type != 'llm':
            raise ValueError(f"Run type {run.run_type} is not supported for dataset of type 'chat'")
        try:
            inputs = {'input': ls_utils.get_messages_from_inputs(run.inputs)}
        except ValueError:
            raise ValueError(f'Error converting LLM run inputs to chat messages for run {run.id} with inputs {run.inputs}')
        if not run.outputs:
            outputs = None
        else:
            try:
                outputs = {'output': ls_utils.get_message_generation_from_outputs(run.outputs)}
            except ValueError:
                raise ValueError(f'Error converting LLM run outputs to chat generations for run {run.id} with outputs {run.outputs}')
    elif dataset_type == ls_schemas.DataType.kv:
        inputs = run.inputs
        outputs = run.outputs
    else:
        raise ValueError(f'Dataset type {dataset_type} not recognized.')
    return self.create_example(inputs=inputs, outputs=outputs, dataset_id=dataset_id, dataset_name=dataset_name, created_at=created_at)