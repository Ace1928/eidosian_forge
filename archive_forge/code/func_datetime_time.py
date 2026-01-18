import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def datetime_time(module):
    if module.__name__ == 'datetime':
        return module.time
    else:
        return module.Time