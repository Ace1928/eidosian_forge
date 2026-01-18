import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def parse_metadata_json(metadata: str, cli_logger, cf, command_arg='--metadata-json') -> Dict[str, str]:
    try:
        metadata = json.loads(metadata)
        if not isinstance(metadata, dict):
            raise ValueError('The format after deserialization is not a dict')
    except Exception as e:
        cli_logger.error('`{}` is not a valid JSON string, detail error:{}', cf.bold(f'{command_arg}={metadata}'), str(e))
        cli_logger.abort('Valid values look like this: `{}`', cf.bold(f"""{command_arg}='{{"key1": "value1", "key2": "value2"}}'"""))
    return metadata