import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
def print_ulm_info(filename, index=None, verbose=False):
    b = ulmopen(filename, 'r')
    if index is None:
        indices = range(len(b))
    else:
        indices = [index]
    print('{0}  (tag: "{1}", {2})'.format(filename, b.get_tag(), plural(len(b), 'item')))
    for i in indices:
        print('item #{0}:'.format(i))
        print(b[i].tostr(verbose))