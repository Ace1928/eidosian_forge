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
def do_test_invalid_host(self):
    self.config.set('Credentials', 'gs_host', INVALID_HOSTNAME_HOST)
    self.config.set('Credentials', 's3_host', INVALID_HOSTNAME_HOST)
    self.assertConnectionThrows(S3Connection, https_connection.InvalidCertificateException)
    self.assertConnectionThrows(GSConnection, https_connection.InvalidCertificateException)