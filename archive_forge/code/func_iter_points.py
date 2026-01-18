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
def iter_points(x):
    """Iterates over a list representing a feature, and returns a list of points,
    whatever the shape of the array (Point, MultiPolyline, etc).
    """
    if isinstance(x, (list, tuple)):
        if len(x):
            if isinstance(x[0], (list, tuple)):
                out = []
                for y in x:
                    out += iter_points(y)
                return out
            else:
                return [x]
        else:
            return []
    else:
        raise ValueError(f'List/tuple type expected. Got {x!r}.')