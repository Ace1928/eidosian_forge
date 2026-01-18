import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def add_meta_spec(klass):
    spec = klass()
    if not hasattr(spec, 'type'):
        name = klass.__name__.replace('MetaSpec_', '')
        spec.type = name
    spec.settable_attributes = set(spec.attributes) | {'time'}
    _META_SPECS[spec.type_byte] = spec
    _META_SPECS[spec.type] = spec
    _META_SPEC_BY_TYPE[spec.type] = spec