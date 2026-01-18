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
def read_run(self, run_id: ID_TYPE, load_child_runs: bool=False) -> ls_schemas.Run:
    """Read a run from the LangSmith API.

        Parameters
        ----------
        run_id : str or UUID
            The ID of the run to read.
        load_child_runs : bool, default=False
            Whether to load nested child runs.

        Returns:
        -------
        Run
            The run.
        """
    response = self.request_with_retries('GET', f'/runs/{_as_uuid(run_id, 'run_id')}')
    run = ls_schemas.Run(**response.json(), _host_url=self._host_url)
    if load_child_runs and run.child_run_ids:
        run = self._load_child_runs(run)
    return run