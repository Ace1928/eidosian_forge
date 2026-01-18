import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts(draft3='date', raises=ValueError)
def is_date(instance):
    if not isinstance(instance, str_types):
        return True
    return datetime.datetime.strptime(instance, '%Y-%m-%d')