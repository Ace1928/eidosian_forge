import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_image_list_shared_member_status_lower(self):
    arglist = ['--shared', '--member-status', 'ALl']
    verifylist = [('visibility', 'shared'), ('long', False), ('member_status', 'all')]
    self.check_parser(self.cmd, arglist, verifylist)