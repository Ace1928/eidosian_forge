import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def entry_from_path_stat(path: str, stat: Stat) -> DirEntry:
    assert not path.endswith('/')
    _, obj = split_path(path)
    name = obj.split('/')[-1]
    return DirEntry(name=name, path=path, is_dir=False, is_file=True, stat=stat)