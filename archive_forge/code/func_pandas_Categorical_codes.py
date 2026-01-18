import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def pandas_Categorical_codes(cat):
    if hasattr(cat, 'cat'):
        cat = cat.cat
    if hasattr(cat, 'codes'):
        return cat.codes
    else:
        return cat.labels