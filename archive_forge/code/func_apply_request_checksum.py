import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def apply_request_checksum(request):
    checksum_context = request.get('context', {}).get('checksum', {})
    algorithm = checksum_context.get('request_algorithm')
    if not algorithm:
        return
    if algorithm == 'conditional-md5':
        conditionally_calculate_md5(request)
    elif algorithm['in'] == 'header':
        _apply_request_header_checksum(request)
    elif algorithm['in'] == 'trailer':
        _apply_request_trailer_checksum(request)
    else:
        raise FlexibleChecksumError(error_msg='Unknown checksum variant: %s' % algorithm['in'])