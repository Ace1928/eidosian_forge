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
def get_cgroupv2_used_memory(stat_file, usage_file):
    inactive_file_bytes = -1
    current_usage = -1
    with open(usage_file, 'r') as f:
        current_usage = int(f.read().strip())
    with open(stat_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'inactive_file' in line:
                inactive_file_bytes = int(line.split()[1])
        if current_usage >= 0 and inactive_file_bytes >= 0:
            working_set = current_usage - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None