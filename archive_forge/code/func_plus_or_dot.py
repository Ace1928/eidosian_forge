import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def plus_or_dot(pieces) -> str:
    """Return a + if we don't already have one, else return a ."""
    if '+' in pieces.get('closest-tag', ''):
        return '.'
    return '+'