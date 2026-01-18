from unittest import mock
from glance.api import property_protections
from glance import context
from glance import gateway
from glance import notifier
from glance import quota
from glance.tests.unit import utils as unit_test_utils
import glance.tests.utils as test_utils
def test_get_repo_member_property(self):
    """Test that the image.member property is propagated all the way from
        the DB to the top of the gateway repo stack.
        """
    db_api = unit_test_utils.FakeDB()
    gw = gateway.Gateway(db_api=db_api)
    ctxt = context.RequestContext(tenant=unit_test_utils.TENANT1)
    repo = gw.get_repo(ctxt)
    image = repo.get(unit_test_utils.UUID1)
    self.assertIsNone(image.member)
    ctxt = context.RequestContext(tenant=unit_test_utils.TENANT2)
    repo = gw.get_repo(ctxt)
    image = repo.get(unit_test_utils.UUID1)
    self.assertEqual(unit_test_utils.TENANT2, image.member)