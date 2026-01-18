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
@patch.object(libcloud.requests.packages.chardet, '__version__', '2.3.0')
def test_init_once_correct_chardet_version(self, *args):
    _init_once()