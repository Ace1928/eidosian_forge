import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
@_checks_drafts(draft3='color', raises=(ValueError, TypeError))
def is_css21_color(instance):
    if not isinstance(instance, str_types) or instance.lower() in webcolors.css21_names_to_hex:
        return True
    return is_css_color_code(instance)