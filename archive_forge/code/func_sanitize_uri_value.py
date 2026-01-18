from itertools import chain
import re
import warnings
from xml.sax.saxutils import unescape
from bleach import html5lib_shim
from bleach import parse_shim
def sanitize_uri_value(self, value, allowed_protocols):
    """Checks a uri value to see if it's allowed

        :arg value: the uri value to sanitize
        :arg allowed_protocols: list of allowed protocols

        :returns: allowed value or None

        """
    normalized_uri = html5lib_shim.convert_entities(value)
    normalized_uri = re.sub('[`\\000-\\040\\177-\\240\\s]+', '', normalized_uri)
    normalized_uri = normalized_uri.replace('�', '')
    normalized_uri = normalized_uri.lower()
    try:
        parsed = parse_shim.urlparse(normalized_uri)
    except ValueError:
        return None
    if parsed.scheme:
        if parsed.scheme in allowed_protocols:
            return value
    else:
        if normalized_uri.startswith('#'):
            return value
        if ':' in normalized_uri and normalized_uri.split(':')[0] in allowed_protocols:
            return value
        if 'http' in allowed_protocols or 'https' in allowed_protocols:
            return value
    return None