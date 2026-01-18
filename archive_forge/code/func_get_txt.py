import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def get_txt(txt, rank):
    if hasattr(txt, 'write'):
        return txt
    elif rank == 0:
        if txt is None:
            return open(os.devnull, 'w')
        elif txt == '-':
            return sys.stdout
        else:
            return open(txt, 'w', 1)
    else:
        return open(os.devnull, 'w')