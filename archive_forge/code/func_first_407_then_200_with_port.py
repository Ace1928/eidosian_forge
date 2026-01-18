from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(netloc='example.com:8000', path='.*/someprotecteurl')
def first_407_then_200_with_port(url, request):
    if request.headers.get('cookie', '').startswith('macaroon-'):
        return {'status_code': 200, 'content': {'Value': 'some value'}}
    else:
        resp = response(status_code=407, content={'Info': {'Macaroon': json_macaroon, 'MacaroonPath': '/', 'CookieNameSuffix': 'test'}, 'Message': 'verification failed: no macaroon cookies in request', 'Code': 'macaroon discharge required'}, headers={'Content-Type': 'application/json'}, request=request)
        return request.hooks['response'][0](resp)