from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@urlmatch(path='.*/discharge')
def discharge_401(url, request):
    return {'status_code': 401, 'content': {'Code': 'interaction required', 'Info': {'VisitURL': 'http://example.com/visit', 'WaitURL': 'http://example.com/wait'}}, 'headers': {'WWW-Authenticate': 'Macaroon'}}