import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def import_datetime(module_type):
    global datetime_module, mxDateTime_module
    module_type = module_type.lower() if module_type else 'datetime'
    if module_type == 'datetime':
        if datetime_module is None:
            import datetime as datetime_module
        return datetime_module
    elif module_type == 'mxdatetime':
        if mxDateTime_module is None:
            from mx import DateTime as mxDateTime_module
        return mxDateTime_module
    else:
        raise ImportError('Invalid datetime module %r' % module_type)