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
def make_stat(item: Mapping[str, Any]) -> Stat:
    if 'metadata' in item and 'blobfile-mtime' in item['metadata']:
        mtime = float(item['metadata']['blobfile-mtime'])
    else:
        mtime = _parse_timestamp(item['updated'])
    return Stat(size=int(item['size']), mtime=mtime, ctime=_parse_timestamp(item['timeCreated']), md5=get_md5(item), version=item['generation'])