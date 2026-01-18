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
def delete_example(self, example_id: ID_TYPE) -> None:
    """Delete an example by ID.

        Parameters
        ----------
        example_id : str or UUID
            The ID of the example to delete.
        """
    response = self.session.delete(f'{self.api_url}/examples/{_as_uuid(example_id, 'example_id')}', headers=self._headers)
    ls_utils.raise_for_status_with_text(response)