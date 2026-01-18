import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_generator_proxy(self):

    def _test_decorator():
        i = 1
        resp = create_response_obj_with_req_id(REQUEST_ID)
        while True:
            yield (i, resp)
            i += 1
    gen_obj = _test_decorator()
    proxy = utils.GeneratorProxy(gen_obj)
    self.assertIsInstance(proxy, type(gen_obj))
    self.assertEqual([], proxy.request_ids)
    proxy.next()
    self.assertEqual([REQUEST_ID], proxy.request_ids)
    proxy.next()
    proxy.next()
    self.assertEqual(1, len(proxy.request_ids))