import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
def test_ChainingRegExpFilter_not_match(self):
    filter_list = [filters.ChainingRegExpFilter('nice', 'root', 'nice', '-?\\d+'), filters.CommandFilter('cat', 'root')]
    args_invalid = (['nice', '5', 'ls', '/a'], ['nice', '--5', 'cat', '/a'], ['nice2', '5', 'cat', '/a'], ['nice', 'cat', '/a'], ['nice', '5'])
    dirs = ['/bin', '/usr/bin']
    for args in args_invalid:
        self.assertRaises(wrapper.NoFilterMatched, wrapper.match_filter, filter_list, args, dirs)