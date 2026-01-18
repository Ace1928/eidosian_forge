import datetime
from unittest import mock
import uuid
from oslo_config import cfg
import oslo_utils.importutils
import glance.async_
from glance.async_ import taskflow_executor
from glance.common import exception
from glance.common import timeutils
from glance import domain
import glance.tests.utils as test_utils
def test_eq_with_uneqal_dict(self):
    a_dict = {'foo': 'bar', 'snitch': 'golden'}
    extra_properties = domain.ExtraProperties(a_dict)
    ref_extra_properties = {'boo': 'far', 'gnitch': 'solden'}
    self.assertNotEqual(ref_extra_properties, extra_properties)