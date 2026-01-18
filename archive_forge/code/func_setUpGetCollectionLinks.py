from unittest import mock
from urllib import parse as urlparse
from heat.api.openstack.v1.views import views_common
from heat.tests import common
def setUpGetCollectionLinks(self):
    self.items = [self.stack1, self.stack2]
    self.request.params = {'limit': '2'}
    self.request.path_url = 'http://example.com/fake/path'