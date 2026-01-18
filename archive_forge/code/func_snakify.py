from __future__ import annotations
import logging  # isort:skip
import re
from typing import Any, Iterable, overload
from urllib.parse import quote_plus
def snakify(name: str, sep: str='_') -> str:
    """ Convert CamelCase to snake_case. """
    name = re.sub('([A-Z]+)([A-Z][a-z])', f'\\1{sep}\\2', name)
    name = re.sub('([a-z\\d])([A-Z])', f'\\1{sep}\\2', name)
    return name.lower()