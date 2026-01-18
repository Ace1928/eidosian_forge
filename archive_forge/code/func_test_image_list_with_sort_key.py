import testtools
from testtools import matchers
from zunclient.tests.unit import utils
from zunclient.v1 import images
def test_image_list_with_sort_key(self):
    expect = [('GET', '/v1/images/?sort_key=image_id', {}, None)]
    self._test_image_list_with_filters(sort_key='image_id', expect=expect)