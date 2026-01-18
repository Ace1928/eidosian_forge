from unittest import mock
from urllib import parse as urlparse
from heat.api.openstack.v1.views import views_common
from heat.tests import common
def test_get_collection_links_handles_invalid_limits(self):
    self.setUpGetCollectionLinks()
    self.request.params = {'limit': 'foo'}
    links = views_common.get_collection_links(self.request, self.items)
    self.assertEqual([], links)
    self.request.params = {'limit': None}
    links = views_common.get_collection_links(self.request, self.items)
    self.assertEqual([], links)