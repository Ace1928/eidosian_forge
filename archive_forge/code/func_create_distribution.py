import os
import io
import sys
import unittest
import warnings
import textwrap
from unittest import mock
from distutils.dist import Distribution, fix_help_options
from distutils.cmd import Command
from test.support import (
from test.support.os_helper import TESTFN
from distutils.tests import support
from distutils import log
def create_distribution(self, configfiles=()):
    d = TestDistribution()
    d._config_files = configfiles
    d.parse_config_files()
    d.parse_command_line()
    return d