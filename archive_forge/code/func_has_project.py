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
def has_project(self, project_name: str, *, project_id: Optional[str]=None) -> bool:
    """Check if a project exists.

        Parameters
        ----------
        project_name : str
            The name of the project to check for.
        project_id : str or None, default=None
            The ID of the project to check for.

        Returns:
        -------
        bool
            Whether the project exists.
        """
    try:
        self.read_project(project_name=project_name)
    except ls_utils.LangSmithNotFoundError:
        return False
    return True