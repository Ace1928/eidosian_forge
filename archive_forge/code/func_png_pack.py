import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def png_pack(png_tag, data):
    chunk_head = png_tag + data
    return struct.pack('!I', len(data)) + chunk_head + struct.pack('!I', 4294967295 & zlib.crc32(chunk_head))