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
@ls_utils.xor_args(('dataset_name', 'dataset_id'))
def read_dataset(self, *, dataset_name: Optional[str]=None, dataset_id: Optional[ID_TYPE]=None) -> ls_schemas.Dataset:
    """Read a dataset from the LangSmith API.

        Parameters
        ----------
        dataset_name : str or None, default=None
            The name of the dataset to read.
        dataset_id : UUID or None, default=None
            The ID of the dataset to read.

        Returns:
        -------
        Dataset
            The dataset.
        """
    path = '/datasets'
    params: Dict[str, Any] = {'limit': 1}
    if dataset_id is not None:
        path += f'/{_as_uuid(dataset_id, 'dataset_id')}'
    elif dataset_name is not None:
        params['name'] = dataset_name
    else:
        raise ValueError('Must provide dataset_name or dataset_id')
    response = self.request_with_retries('GET', path, params=params)
    result = response.json()
    if isinstance(result, list):
        if len(result) == 0:
            raise ls_utils.LangSmithNotFoundError(f'Dataset {dataset_name} not found')
        return ls_schemas.Dataset(**result[0], _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())
    return ls_schemas.Dataset(**result, _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())