import operator
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ._parser import (
from ._parser import (
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name
def format_full_version(info: 'sys._version_info') -> str:
    version = '{0.major}.{0.minor}.{0.micro}'.format(info)
    kind = info.releaselevel
    if kind != 'final':
        version += kind[0] + str(info.serial)
    return version