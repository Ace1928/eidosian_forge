import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts(draft3='time', raises=ValueError)
def is_time(instance):
    if not isinstance(instance, str_types):
        return True
    return datetime.datetime.strptime(instance, '%H:%M:%S')