import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def md5_string(string: str) -> B64MD5:
    return _b64_from_hasher(_md5(string.encode('utf-8')))