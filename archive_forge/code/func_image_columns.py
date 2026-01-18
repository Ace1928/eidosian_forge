from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_image
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def image_columns(self, image):
    columnlist = ('id', 'name', 'owner', 'protected', 'status', 'tags', 'visibility')
    return columnlist