import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def md5_file_b64(*paths: StrPath) -> B64MD5:
    return _b64_from_hasher(_md5_file_hasher(*paths))