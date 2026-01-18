from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test__obj_to_readable_str():

    def t(obj, expected):
        got = _obj_to_readable_str(obj)
        assert type(got) is str
        assert got == expected
    t(1, '1')
    t(1.0, '1.0')
    t('asdf', 'asdf')
    t(six.u('asdf'), 'asdf')
    if sys.version_info >= (3,):
        t('€'.encode('utf-8'), six.u('€'))
        t('€'.encode('iso-8859-15'), "b'\\xa4'")
    else:
        t(six.u('€'), "u'\\u20ac'")