import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def check_unmatched(unmatched):
    found = re.search('\\w+', unmatched)
    if found:
        raise ValueError(f'Unexpected {found.group(0)!r}')