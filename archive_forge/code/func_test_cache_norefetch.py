import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
from httmock import HTTMock, urlmatch
def test_cache_norefetch(self):
    key = bakery.generate_key()

    @urlmatch(path='.*/discharge/info')
    def discharge_info(url, request):
        return {'status_code': 200, 'content': {'Version': bakery.LATEST_VERSION, 'PublicKey': str(key.public_key)}}
    expectInfo = bakery.ThirdPartyInfo(public_key=key.public_key, version=bakery.LATEST_VERSION)
    kr = httpbakery.ThirdPartyLocator(allow_insecure=True)
    with HTTMock(discharge_info):
        info = kr.third_party_info('http://0.1.2.3/')
    self.assertEqual(info, expectInfo)
    info = kr.third_party_info('http://0.1.2.3/')
    self.assertEqual(info, expectInfo)