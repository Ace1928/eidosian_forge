import io
import logging
import os
import pprint
import sys
import typing as ty
import fixtures
from oslotest import base
import testtools.content
from openstack.tests import fixtures as os_fixtures
from openstack import utils
def attachLogs(self, *args):

    def reader():
        self._log_stream.seek(0)
        while True:
            x = self._log_stream.read(4096)
            if not x:
                break
            yield x.encode('utf8')
    content = testtools.content.content_from_reader(reader, testtools.content_type.UTF8_TEXT, False)
    self.addDetail('logging', content)