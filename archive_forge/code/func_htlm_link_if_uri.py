from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
def htlm_link_if_uri(value):
    try:
        uri = value.uri
        return '<a href="%s">%s</a>' % (uri, str(value))
    except AttributeError:
        return str(value)