import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
def get_tracer_project(return_default_value=True) -> Optional[str]:
    """Get the project name for a LangSmith tracer."""
    return os.environ.get('HOSTED_LANGSERVE_PROJECT_NAME', get_env_var('PROJECT', default=get_env_var('SESSION', default='default' if return_default_value else None)))