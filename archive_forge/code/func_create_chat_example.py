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
def create_chat_example(self, messages: List[Union[Mapping[str, Any], ls_schemas.BaseMessageLike]], generations: Optional[Union[Mapping[str, Any], ls_schemas.BaseMessageLike]]=None, dataset_id: Optional[ID_TYPE]=None, dataset_name: Optional[str]=None, created_at: Optional[datetime.datetime]=None) -> ls_schemas.Example:
    """Add an example (row) to a Chat-type dataset."""
    final_input = []
    for message in messages:
        if ls_utils.is_base_message_like(message):
            final_input.append(ls_utils.convert_langchain_message(cast(ls_schemas.BaseMessageLike, message)))
        else:
            final_input.append(cast(dict, message))
    final_generations = None
    if generations is not None:
        if ls_utils.is_base_message_like(generations):
            final_generations = ls_utils.convert_langchain_message(cast(ls_schemas.BaseMessageLike, generations))
        else:
            final_generations = cast(dict, generations)
    return self.create_example(inputs={'input': final_input}, outputs={'output': final_generations} if final_generations is not None else None, dataset_id=dataset_id, dataset_name=dataset_name, created_at=created_at)