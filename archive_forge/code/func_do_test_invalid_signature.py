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
def do_test_invalid_signature(self):
    self.config.set('Boto', 'ca_certificates_file', DEFAULT_CA_CERTS_FILE)
    self.assertConnectionThrows(S3Connection, ssl.SSLError)
    self.assertConnectionThrows(GSConnection, ssl.SSLError)