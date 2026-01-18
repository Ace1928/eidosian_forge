import errno
import numbers
import os
import subprocess
import sys
from itertools import zip_longest
from io import UnsupportedOperation
def close_open_fds(keep=None):
    keep = list(uniq(sorted((f for f in map(maybe_fileno, keep or []) if f is not None))))
    maxfd = get_fdmax(default=2048)
    kL, kH = (iter([-1] + keep), iter(keep + [maxfd]))
    for low, high in zip_longest(kL, kH):
        if low + 1 != high:
            closerange(low + 1, high)