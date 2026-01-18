from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_expansion(params, expansion):
    """Append EXPANSION to params."""
    if expansion is not None:
        params.extend(['EXPANSION', expansion])