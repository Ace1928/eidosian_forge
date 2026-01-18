from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.bakery as bakery
def test_from_dict_lower_case_fields(self):
    err = httpbakery.Error.from_dict({'message': 'm', 'code': 'c'})
    self.assertEqual(err, httpbakery.Error(code='c', message='m', info=None, version=bakery.LATEST_VERSION))