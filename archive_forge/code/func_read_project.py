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
@ls_utils.xor_args(('project_id', 'project_name'))
def read_project(self, *, project_id: Optional[str]=None, project_name: Optional[str]=None, include_stats: bool=False) -> ls_schemas.TracerSessionResult:
    """Read a project from the LangSmith API.

        Parameters
        ----------
        project_id : str or None, default=None
            The ID of the project to read.
        project_name : str or None, default=None
            The name of the project to read.
                Note: Only one of project_id or project_name may be given.
        include_stats : bool, default=False
            Whether to include a project's aggregate statistics in the response.

        Returns:
        -------
        TracerSessionResult
            The project.
        """
    path = '/sessions'
    params: Dict[str, Any] = {'limit': 1}
    if project_id is not None:
        path += f'/{_as_uuid(project_id, 'project_id')}'
    elif project_name is not None:
        params['name'] = project_name
    else:
        raise ValueError('Must provide project_name or project_id')
    params['include_stats'] = include_stats
    response = self.request_with_retries('GET', path, params=params)
    result = response.json()
    if isinstance(result, list):
        if len(result) == 0:
            raise ls_utils.LangSmithNotFoundError(f'Project {project_name} not found')
        return ls_schemas.TracerSessionResult(**result[0], _host_url=self._host_url)
    return ls_schemas.TracerSessionResult(**response.json(), _host_url=self._host_url)