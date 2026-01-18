from __future__ import annotations
import warnings
from typing import Any
def parse_constants_2018toXXXX(d: str) -> dict[str, tuple[float, str, float]]:
    constants = {}
    for line in d.split('\n'):
        name = line[:60].rstrip()
        val = float(line[60:85].replace(' ', '').replace('...', ''))
        uncert = float(line[85:110].replace(' ', '').replace('(exact)', '0'))
        units = line[110:].rstrip()
        constants[name] = (val, units, uncert)
    return constants