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
def get_cgroupv1_used_memory(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        cache_bytes = -1
        rss_bytes = -1
        inactive_file_bytes = -1
        working_set = -1
        for line in lines:
            if 'total_rss ' in line:
                rss_bytes = int(line.split()[1])
            elif 'cache ' in line:
                cache_bytes = int(line.split()[1])
            elif 'inactive_file' in line:
                inactive_file_bytes = int(line.split()[1])
        if cache_bytes >= 0 and rss_bytes >= 0 and (inactive_file_bytes >= 0):
            working_set = rss_bytes + cache_bytes - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None