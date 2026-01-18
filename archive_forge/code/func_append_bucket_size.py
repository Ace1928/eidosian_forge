from redis._parsers.helpers import bool_ok
from ..helpers import get_protocol_version, parse_to_list
from .commands import *  # noqa
from .info import BFInfo, CFInfo, CMSInfo, TDigestInfo, TopKInfo
@staticmethod
def append_bucket_size(params, bucket_size):
    """Append BUCKETSIZE to params."""
    if bucket_size is not None:
        params.extend(['BUCKETSIZE', bucket_size])