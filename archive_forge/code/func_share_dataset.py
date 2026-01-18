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
def share_dataset(self, dataset_id: Optional[ID_TYPE]=None, *, dataset_name: Optional[str]=None) -> ls_schemas.DatasetShareSchema:
    """Get a share link for a dataset."""
    if dataset_id is None and dataset_name is None:
        raise ValueError('Either dataset_id or dataset_name must be given')
    if dataset_id is None:
        dataset_id = self.read_dataset(dataset_name=dataset_name).id
    data = {'dataset_id': str(dataset_id)}
    response = self.session.put(f'{self.api_url}/datasets/{_as_uuid(dataset_id, 'dataset_id')}/share', headers=self._headers, json=data)
    ls_utils.raise_for_status_with_text(response)
    d: dict = response.json()
    return cast(ls_schemas.DatasetShareSchema, {**d, 'url': f'{self._host_url}/public/{d['share_token']}/d'})