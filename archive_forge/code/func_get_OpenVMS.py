import collections
import os
import re
import sys
import functools
import itertools
def get_OpenVMS():
    try:
        import vms_lib
    except ImportError:
        pass
    else:
        csid, cpu_number = vms_lib.getsyi('SYI$_CPU', 0)
        return 'Alpha' if cpu_number >= 128 else 'VAX'