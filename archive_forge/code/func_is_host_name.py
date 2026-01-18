import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts(draft3='host-name', draft4='hostname')
def is_host_name(instance):
    if not isinstance(instance, str_types):
        return True
    if not _host_name_re.match(instance):
        return False
    components = instance.split('.')
    for component in components:
        if len(component) > 63:
            return False
    return True