from unittest import mock
from urllib import parse as urlparse
from heat.api.openstack.v1.views import views_common
from heat.tests import common
def test_get_collection_links_doesnt_create_next_if_no_limit(self):
    self.setUpGetCollectionLinks()
    del self.request.params['limit']
    links = views_common.get_collection_links(self.request, self.items)
    self.assertEqual([], links)