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
def read_dataset_openai_finetuning(self, dataset_id: Optional[str]=None, *, dataset_name: Optional[str]=None) -> list:
    """Download a dataset in OpenAI Jsonl format and load it as a list of dicts.

        Parameters
        ----------
        dataset_id : str
            The ID of the dataset to download.
        dataset_name : str
            The name of the dataset to download.

        Returns:
        -------
        list
            The dataset loaded as a list of dicts.
        """
    path = '/datasets'
    if dataset_id is not None:
        pass
    elif dataset_name is not None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    else:
        raise ValueError('Must provide dataset_name or dataset_id')
    response = self.request_with_retries('GET', f'{path}/{_as_uuid(dataset_id, 'dataset_id')}/openai_ft')
    dataset = [json.loads(line) for line in response.text.strip().split('\n')]
    return dataset