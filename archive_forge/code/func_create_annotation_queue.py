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
def create_annotation_queue(self, *, name: str, description: Optional[str]=None, queue_id: Optional[ID_TYPE]=None) -> ls_schemas.AnnotationQueue:
    """Create an annotation queue on the LangSmith API.

        Args:
            name : str
                The name of the annotation queue.
            description : str, optional
                The description of the annotation queue.
            queue_id : str or UUID, optional
                The ID of the annotation queue.

        Returns:
            AnnotationQueue
                The created annotation queue object.
        """
    body = {'name': name, 'description': description, 'id': queue_id}
    response = self.request_with_retries('POST', '/annotation-queues', json={k: v for k, v in body.items() if v is not None})
    ls_utils.raise_for_status_with_text(response)
    return ls_schemas.AnnotationQueue(**response.json(), _host_url=self._host_url, _tenant_id=self._get_optional_tenant_id())