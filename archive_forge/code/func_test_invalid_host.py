import os
import ssl
import unittest
import mock
from nose.plugins.attrib import attr
import boto
from boto.pyami.config import Config
from boto import exception, https_connection
from boto.gs.connection import GSConnection
from boto.s3.connection import S3Connection
def test_invalid_host(self):
    self.do_test_invalid_host()