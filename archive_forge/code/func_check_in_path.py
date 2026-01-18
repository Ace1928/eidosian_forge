import collections
import copy
import itertools
import random
import re
import warnings
def check_in_path(v):
    if match(v):
        path.append(v)
        return True
    elif v.is_terminal():
        return False
    for child in v:
        if check_in_path(child):
            path.append(v)
            return True
    return False