import unittest
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
from httmock import HTTMock, urlmatch
def test_allow_insecure(self):
    kr = httpbakery.ThirdPartyLocator()
    with self.assertRaises(bakery.ThirdPartyInfoNotFound):
        kr.third_party_info('http://0.1.2.3/')