import time
from unittest import mock
import uuid
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib.tests import fakes
from osc_lib.tests import utils as test_utils
from osc_lib import utils
def test_get_password_bad_once(self):
    answers = [PASSWORD, WASSPORD, DROWSSAP, DROWSSAP]
    with mock.patch('getpass.getpass', side_effect=answers):
        mock_stdin = mock.Mock()
        mock_stdin.isatty = mock.Mock()
        mock_stdin.isatty.return_value = True
        self.assertEqual(DROWSSAP, utils.get_password(mock_stdin))