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
@ls_utils.xor_args(('dataset_id', 'dataset_name'))
def create_example(self, inputs: Mapping[str, Any], dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, created_at: Optional[datetime.datetime]=None, outputs: Optional[Mapping[str, Any]]=None, metadata: Optional[Mapping[str, Any]]=None, example_id: Optional[ID_TYPE]=None) -> ls_schemas.Example:
    """Create a dataset example in the LangSmith API.

        Examples are rows in a dataset, containing the inputs
        and expected outputs (or other reference information)
        for a model or chain.

        Args:
            inputs : Mapping[str, Any]
                The input values for the example.
            dataset_id : UUID or None, default=None
                The ID of the dataset to create the example in.
            dataset_name : str or None, default=None
                The name of the dataset to create the example in.
            created_at : datetime or None, default=None
                The creation timestamp of the example.
            outputs : Mapping[str, Any] or None, default=None
                The output values for the example.
            metadata : Mapping[str, Any] or None, default=None
                The metadata for the example.
            exemple_id : UUID or None, default=None
                The ID of the example to create. If not provided, a new
                example will be created.

        Returns:
            Example: The created example.
        """
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    data = {'inputs': inputs, 'outputs': outputs, 'dataset_id': dataset_id, 'metadata': metadata}
    if created_at:
        data['created_at'] = created_at.isoformat()
    if example_id:
        data['id'] = example_id
    example = ls_schemas.ExampleCreate(**data)
    response = self.session.post(f'{self.api_url}/examples', headers={**self._headers, 'Content-Type': 'application/json'}, data=example.json())
    ls_utils.raise_for_status_with_text(response)
    result = response.json()
    return ls_schemas.Example(**result, _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())