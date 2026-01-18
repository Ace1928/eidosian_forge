import os
import warnings
import re
def lineno_sort_key(entry):
    if 'lineno' in entry:
        return (0, entry['lineno'])
    else:
        return (1, 0)