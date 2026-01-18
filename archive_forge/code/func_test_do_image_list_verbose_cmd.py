import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def test_do_image_list_verbose_cmd(self):
    self._run_command('--os-image-api-version 2 --verbose image-list')
    utils.print_list.assert_called_once_with(mock.ANY, ['ID', 'Name', 'Disk_format', 'Container_format', 'Size', 'Status', 'Owner'])