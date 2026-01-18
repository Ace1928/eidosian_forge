from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import time
import six
from six.moves import input
import boto
import sys
import gslib
from gslib import command_runner
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.command_runner import CommandRunner
from gslib.command_runner import HandleArgCoding
from gslib.command_runner import HandleHeaderCoding
from gslib.exception import CommandException
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import CloudOrLocalObjectCompleter
from gslib.tab_complete import LocalObjectCompleter
from gslib.tab_complete import LocalObjectOrCannedACLCompleter
from gslib.tab_complete import NoOpCompleter
import gslib.tests.testcase as testcase
import gslib.tests.util as util
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils.constants import GSUTIL_PUB_TARBALL
from gslib.utils.text_util import InsistAscii
from gslib.utils.unit_util import SECONDS_PER_DAY
from six import add_move, MovedModule
from six.moves import mock
@unittest.skipIf(util.HAS_NON_DEFAULT_GS_HOST, SKIP_BECAUSE_RETRIES_ARE_SLOW)
def test_no_tracker_file_version_old(self):
    """Tests when no timestamp file exists and VERSION file is old."""
    if os.path.exists(self.timestamp_file_path):
        os.remove(self.timestamp_file_path)
    self.assertFalse(os.path.exists(self.timestamp_file_path))
    self.version_mod_time = 0
    expected = not self._IsPackageOrCloudSDKInstall()
    self.assertEqual(expected, self.command_runner.MaybeCheckForAndOfferSoftwareUpdate('ls', 0))