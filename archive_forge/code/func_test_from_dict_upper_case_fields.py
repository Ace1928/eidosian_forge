from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.bakery as bakery
def test_from_dict_upper_case_fields(self):
    err = httpbakery.Error.from_dict({'Message': 'm', 'Code': 'c'})
    self.assertEqual(err, httpbakery.Error(code='c', message='m', info=None, version=bakery.LATEST_VERSION))