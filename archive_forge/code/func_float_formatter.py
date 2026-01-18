from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def float_formatter(f: Union[float, int]) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f:
        return ''
    if isinstance(f, int):
        return str(f)
    if f >= 1000:
        s = f'{f:.0f}'
    else:
        s = f'{f:.4g}'
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s