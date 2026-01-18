import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def test_image_member_get(self):
    image_member = self.image_member_repo_proxy.get(TENANT1)
    self.assertIsInstance(image_member, glance.notifier.ImageMemberProxy)
    self.assertEqual('image_member_from_get', image_member.repo)