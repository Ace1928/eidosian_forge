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
@unittest.skipUnless(ARGCOMPLETE_AVAILABLE, 'Tab completion requires argcomplete')
def test_command_argument_parser_setup_nested_argparse_arguments(self):
    command_map = {FakeCommandWithNestedArguments.command_spec.command_name: FakeCommandWithNestedArguments()}
    runner = CommandRunner(bucket_storage_uri_class=self.mock_bucket_storage_uri, gsutil_api_class_map_factory=self.mock_gsutil_api_class_map_factory, command_map=command_map)
    parser = FakeArgparseParser()
    runner.ConfigureCommandArgumentParsers(parser)
    cur_subparser = parser.subparsers.parsers[0]
    self.assertEqual(cur_subparser.name, FakeCommandWithNestedArguments.command_spec.command_name)
    cur_subparser = cur_subparser.subparsers.parsers[0]
    self.assertEqual(cur_subparser.name, FakeCommandWithNestedArguments.subcommand_name)
    cur_subparser = cur_subparser.subparsers.parsers[0]
    self.assertEqual(cur_subparser.name, FakeCommandWithNestedArguments.subcommand_subname)
    self.assertEqual(len(cur_subparser.arguments), FakeCommandWithNestedArguments.num_args)