from __future__ import absolute_import
from decimal import Decimal
from .errors import JSONDecodeError
from .raw_json import RawJSON
from .decoder import JSONDecoder
from .encoder import JSONEncoder, JSONEncoderForHTML
def simple_first(kv):
    """Helper function to pass to item_sort_key to sort simple
    elements to the top, then container elements.
    """
    return (isinstance(kv[1], (list, dict, tuple)), kv[0])