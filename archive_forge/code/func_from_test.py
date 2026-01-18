from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
@classmethod
def from_test(cls, client: Optional[ls_client.Client], func: Callable) -> _LangSmithTestSuite:
    client = client or ls_client.Client()
    test_suite_name = _get_test_suite_name(func)
    with cls._lock:
        if not cls._instances:
            cls._instances = {}
        if test_suite_name not in cls._instances:
            test_suite = _get_test_suite(client, test_suite_name)
            experiment = _start_experiment(client, test_suite)
            cls._instances[test_suite_name] = cls(client, experiment, test_suite)
    return cls._instances[test_suite_name]