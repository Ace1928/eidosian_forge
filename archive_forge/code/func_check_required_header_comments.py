from __future__ import annotations
import io
import os
import re
import subprocess
import sys
import tempfile
from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate
def check_required_header_comments():
    if 'PS-Adobe' not in self.info:
        msg = 'EPS header missing "%!PS-Adobe" comment'
        raise SyntaxError(msg)
    if 'BoundingBox' not in self.info:
        msg = 'EPS header missing "%%BoundingBox" comment'
        raise SyntaxError(msg)