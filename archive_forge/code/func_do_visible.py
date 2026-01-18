from unittest import mock
from oslo_config import cfg
from glance import context
from glance.tests.unit import utils as unit_utils
from glance.tests import utils
def do_visible(self, exp_res, img_owner, img_public, **kwargs):
    """
        Perform a context visibility test.  Creates a (fake) image
        with the specified owner and is_public attributes, then
        creates a context with the given keyword arguments and expects
        exp_res as the result of an is_image_visible() call on the
        context.
        """
    img = _fake_image(img_owner, img_public)
    ctx = context.RequestContext(**kwargs)
    self.assertEqual(exp_res, self.db_api.is_image_visible(ctx, img))