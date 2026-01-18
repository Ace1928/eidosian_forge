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
def add_runs_to_annotation_queue(self, queue_id: ID_TYPE, *, run_ids: List[ID_TYPE]) -> None:
    """Add runs to an annotation queue with the specified queue ID.

        Args:
            queue_id (ID_TYPE): The ID of the annotation queue.
            run_ids (List[ID_TYPE]): The IDs of the runs to be added to the annotation
                queue.
        """
    response = self.request_with_retries('POST', f'/annotation-queues/{_as_uuid(queue_id, 'queue_id')}/runs', json=[str(_as_uuid(id_, f'run_ids[{i}]')) for i, id_ in enumerate(run_ids)])
    ls_utils.raise_for_status_with_text(response)