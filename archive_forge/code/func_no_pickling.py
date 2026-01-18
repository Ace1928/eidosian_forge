import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def no_pickling(*args, **kwargs):
    raise NotImplementedError('Sorry, pickling not yet supported. See https://github.com/pydata/patsy/issues/26 if you want to help.')