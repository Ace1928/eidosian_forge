import os
import json
import time
import base64
import requests
from libcloud.common.base import JsonResponse, ConnectionKey
def is_jwt_expired(jwt):
    """
    Check if jwt is expired

    :param jwt: jwt token to validate expiration
    :type  jwt: str

    :rtype: bool
    """
    jwt = jwt.encode('utf-8')
    signing_input, _ = jwt.rsplit(b'.', 1)
    _, claims_segment = signing_input.split(b'.', 1)
    claimsdata = base64url_decode(claims_segment)
    if isinstance(claimsdata, bytes):
        claimsdata = claimsdata.decode('utf-8')
    data = json.loads(claimsdata)
    return data['exp'] < time.time() + 60