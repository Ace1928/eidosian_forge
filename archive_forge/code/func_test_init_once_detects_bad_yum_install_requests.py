import os
import sys
import logging
import tempfile
from unittest.mock import patch
import libcloud
from libcloud import _init_once
from libcloud.base import DriverTypeNotFoundError
from libcloud.test import unittest
from libcloud.utils.loggingconnection import LoggingConnection
@patch.object(libcloud.requests, '__version__', '2.6.0')
@patch.object(libcloud.requests.packages.chardet, '__version__', '2.2.1')
def test_init_once_detects_bad_yum_install_requests(self, *args):
    expected_msg = 'Known bad version of requests detected'
    with self.assertRaisesRegex(AssertionError, expected_msg):
        _init_once()