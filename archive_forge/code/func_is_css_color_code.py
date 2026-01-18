import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
def is_css_color_code(instance):
    return webcolors.normalize_hex(instance)