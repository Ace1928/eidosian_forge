import base64
import hashlib
import mmap
import os
import sys
from pathlib import Path
from typing import NewType, Union
from wandb.sdk.lib.paths import StrPath
def b64_to_hex_id(string: B64MD5) -> HexMD5:
    return HexMD5(base64.standard_b64decode(string).hex())