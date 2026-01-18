from unittest import mock
from glance.api import property_protections
from glance import context
from glance import gateway
from glance import notifier
from glance import quota
from glance.tests.unit import utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_property_repo(self):
    repo = self.gateway.get_metadef_property_repo(self.context)
    self.assertIsInstance(repo, notifier.MetadefPropertyRepoProxy)