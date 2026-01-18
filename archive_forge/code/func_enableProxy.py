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
def enableProxy(self):
    self.config.set('Boto', 'proxy', PROXY_HOST)
    self.config.set('Boto', 'proxy_port', PROXY_PORT)