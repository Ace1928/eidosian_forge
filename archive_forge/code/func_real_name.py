from __future__ import annotations # isort:skip
import hashlib
import json
from os.path import splitext
from pathlib import Path
from sys import stdout
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import urljoin
from urllib.request import urlopen
def real_name(name: str) -> str:
    real_name, ext = splitext(name)
    if ext == '.zip':
        if not splitext(real_name)[1]:
            return f'{real_name}.csv'
        else:
            return real_name
    else:
        return name