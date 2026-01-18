import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts('ipv6', raises=socket.error)
def is_ipv6(instance):
    if not isinstance(instance, str_types):
        return True
    return socket.inet_pton(socket.AF_INET6, instance)