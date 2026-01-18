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
def read_example(self, example_id: ID_TYPE, *, as_of: Optional[datetime.datetime]=None) -> ls_schemas.Example:
    """Read an example from the LangSmith API.

        Args:
            example_id (UUID): The ID of the example to read.

        Returns:
            Example: The example.
        """
    response = self.request_with_retries('GET', f'/examples/{_as_uuid(example_id, 'example_id')}', params={'as_of': as_of.isoformat() if as_of else None})
    return ls_schemas.Example(**response.json(), _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())