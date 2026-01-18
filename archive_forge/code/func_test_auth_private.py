from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def test_auth_private(self):
    """
        Tests that an authenticated context (with is_admin set to
        False) can access an image with is_public set to False.
        """
    self.do_visible(True, None, False, project_id='froggy')