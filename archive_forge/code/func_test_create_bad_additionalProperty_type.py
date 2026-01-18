import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_create_bad_additionalProperty_type(self):
    properties = {'name': 'image-1', 'bad_prop': True}
    with testtools.ExpectedException(TypeError):
        self.controller.create(**properties)