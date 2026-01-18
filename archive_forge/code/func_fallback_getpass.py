import contextlib
import io
import os
import sys
import warnings
def fallback_getpass(prompt='Password: ', stream=None):
    warnings.warn('Can not control echo on the terminal.', GetPassWarning, stacklevel=2)
    if not stream:
        stream = sys.stderr
    print('Warning: Password input may be echoed.', file=stream)
    return _raw_input(prompt, stream)