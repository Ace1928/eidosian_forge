import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_format_size(self):
    self.assertEqual('999', utils.format_size(999))
    self.assertEqual('100K', utils.format_size(100000))
    self.assertEqual('2M', utils.format_size(2000000))
    self.assertEqual('16.4M', utils.format_size(16361280))
    self.assertEqual('1.6G', utils.format_size(1576395005))
    self.assertEqual('0', utils.format_size(None))