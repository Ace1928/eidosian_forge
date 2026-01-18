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
def create_llm_example(self, prompt: str, generation: Optional[str]=None, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, created_at: Optional[datetime.datetime]=None) -> ls_schemas.Example:
    """Add an example (row) to an LLM-type dataset."""
    return self.create_example(inputs={'input': prompt}, outputs={'output': generation}, dataset_id=dataset_id, dataset_name=dataset_name, created_at=created_at)