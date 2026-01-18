from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def ratelimited(*args, **kwargs):
    if sys.version_info >= (3, 8):
        real_time = time.process_time
    else:
        real_time = time.clock
    if minrate is not None:
        elapsed = real_time() - last[0]
        left = minrate - elapsed
        if left > 0:
            time.sleep(left)
        last[0] = real_time()
    ret = f(*args, **kwargs)
    return ret