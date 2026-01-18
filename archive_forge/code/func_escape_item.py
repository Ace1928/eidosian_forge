import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_item(val, charset, mapping=None):
    if mapping is None:
        mapping = encoders
    encoder = mapping.get(type(val))
    if not encoder:
        try:
            encoder = mapping[str]
        except KeyError:
            raise TypeError('no default type converter defined')
    if encoder in (escape_dict, escape_sequence):
        val = encoder(val, charset, mapping)
    else:
        val = encoder(val, mapping)
    return val