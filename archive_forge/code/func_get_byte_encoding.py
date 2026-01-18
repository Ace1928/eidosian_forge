from __future__ import annotations
import re
import typing
import warnings
import wcwidth
def get_byte_encoding() -> Literal['utf8', 'narrow', 'wide']:
    return _byte_encoding