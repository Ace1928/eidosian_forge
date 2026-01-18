import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
from httmock import HTTMock, urlmatch
@urlmatch(path='.*/discharge/info')
def discharge_info(url, request):
    return {'status_code': 404}