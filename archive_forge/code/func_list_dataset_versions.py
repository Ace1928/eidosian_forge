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
def list_dataset_versions(self, *, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, search: Optional[str]=None, limit: Optional[int]=None) -> Iterator[ls_schemas.DatasetVersion]:
    """List dataset versions.

        Args:
            dataset_id (Optional[ID_TYPE]): The ID of the dataset.
            dataset_name (Optional[str]): The name of the dataset.
            search (Optional[str]): The search query.
            limit (Optional[int]): The maximum number of versions to return.

        Returns:
            Iterator[ls_schemas.DatasetVersion]: An iterator of dataset versions.
        """
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    params = {'search': search, 'limit': min(limit, 100) if limit is not None else 100}
    for i, version in enumerate(self._get_paginated_list(f'/datasets/{_as_uuid(dataset_id, 'dataset_id')}/versions', params=params)):
        yield ls_schemas.DatasetVersion(**version)
        if limit is not None and i + 1 >= limit:
            break