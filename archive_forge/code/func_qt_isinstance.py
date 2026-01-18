from __future__ import print_function, absolute_import
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
from shibokensupport.signature.mapping import update_mapping, namespace
from textwrap import dedent
def qt_isinstance(inst, the_type):
    if the_type == float:
        return isinstance(inst, int) or isinstance(int, float)
    try:
        return isinstance(inst, the_type)
    except TypeError as e:
        print('FIXME', e)
        return False