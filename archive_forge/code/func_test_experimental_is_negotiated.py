import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
@ddt.data(*combos)
@ddt.unpack
def test_experimental_is_negotiated(self, cache, multistore):
    self.config(enabled_backends=multistore)
    self.config(image_cache_dir=cache)
    to_check = self._get_list_of_version_ids('EXPERIMENTAL')
    for version_id in to_check:
        self._assert_version_is_negotiated(version_id)