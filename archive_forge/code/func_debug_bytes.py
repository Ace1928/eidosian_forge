import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def debug_bytes(*args) -> str:
    index = range(max(map(len, args)))
    result = []
    for arg in [index] + list(args) + [[int(a != b) for a, b in zip(args[-1], args[-2])]]:
        result.append(' '.join((f'{x:03}' for x in arg)))
    return 'bytes mismatch\n' + '\n'.join(result)