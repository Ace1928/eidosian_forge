from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_items(params, items):
    """Append ITEMS to params."""
    params.extend(['ITEMS'])
    params += items