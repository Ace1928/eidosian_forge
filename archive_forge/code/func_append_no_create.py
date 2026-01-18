from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_no_create(params, noCreate):
    """Append NOCREATE tag to params."""
    if noCreate is not None:
        params.extend(['NOCREATE'])