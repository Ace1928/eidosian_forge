from __future__ import annotations
import binascii
import collections
import concurrent.futures
import contextlib
import hashlib
import io
import itertools
import math
import multiprocessing as mp
import os
import re
import shutil
import stat as stat_module
import tempfile
import time
import urllib.parse
from functools import partial
from types import ModuleType
from typing import (
import urllib3
import filelock
from blobfile import _azure as azure
from blobfile import _common as common
from blobfile import _gcp as gcp
from blobfile._common import (
def request_generator():
    account, container, blob = azure.split_path(path)
    for entry in azure.list_blobs(self._conf, path):
        entry_slash_path = _get_slash_path(entry)
        entry_account, entry_container, entry_blob = azure.split_path(entry_slash_path)
        assert entry_account == account and entry_container == container and entry_blob.startswith(blob)
        req = Request(url=azure.build_url(account, '/{container}/{blob}', container=container, blob=entry_blob), method='DELETE', success_codes=(202, 404))
        yield req