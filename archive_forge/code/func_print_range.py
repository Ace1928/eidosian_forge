import dataclasses
import io
import math
from typing import Iterable, Optional
from ortools.math_opt.python import model
def print_range(prefix: str, value: Optional[Range]) -> None:
    buf.write(prefix)
    if value is None:
        buf.write('no finite values')
        return
    buf.write(f'[{value.minimum:<9.2e}, {value.maximum:<9.2e}]')