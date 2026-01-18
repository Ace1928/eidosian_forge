import base64
import io
import logging
from binascii import crc32
from hashlib import sha1, sha256
from botocore.compat import HAS_CRT
from botocore.exceptions import (
from botocore.response import StreamingBody
from botocore.utils import (
def resolve_response_checksum_algorithms(request, operation_model, params, supported_algorithms=None):
    http_checksum = operation_model.http_checksum
    mode_member = http_checksum.get('requestValidationModeMember')
    if mode_member and mode_member in params:
        if supported_algorithms is None:
            supported_algorithms = _SUPPORTED_CHECKSUM_ALGORITHMS
        response_algorithms = {a.lower() for a in http_checksum.get('responseAlgorithms', [])}
        usable_algorithms = []
        for algorithm in _ALGORITHMS_PRIORITY_LIST:
            if algorithm not in response_algorithms:
                continue
            if algorithm in supported_algorithms:
                usable_algorithms.append(algorithm)
        checksum_context = request['context'].get('checksum', {})
        checksum_context['response_algorithms'] = usable_algorithms
        request['context']['checksum'] = checksum_context