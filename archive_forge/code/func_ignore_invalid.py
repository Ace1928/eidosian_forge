from contextlib import contextmanager
import numpy as np
import pytest
import shapely
@contextmanager
def ignore_invalid(condition=True):
    if condition:
        with np.errstate(invalid='ignore'):
            yield
    else:
        yield