import os
import json
import time
import base64
import requests
from libcloud.common.base import JsonResponse, ConnectionKey
def refresh_jwt(jwt):
    """
    Refresh jwt

    :param jwt: jwt token to refresh
    :type  jwt: str

    :rtype: str
    """
    url = IYO_URL + '/v1/oauth/jwt/refresh'
    headers = {'Authorization': 'bearer {}'.format(jwt)}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text