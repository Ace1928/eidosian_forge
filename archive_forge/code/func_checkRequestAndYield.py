import os
import socket
import sys
import time
from breezy import config, controldir, errors, tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.osutils import lexists
from breezy.tests import TestCase, TestCaseWithTransport, TestSkipped, features
from breezy.tests.http_server import HttpServer
def checkRequestAndYield(self, expected, data, offsets):
    self.requireFeature(features.paramiko)
    helper = _mod_sftp._SFTPReadvHelper(offsets, 'artificial_test', _null_report_activity)
    data_f = ReadvFile(data)
    result = list(helper.request_and_yield_offsets(data_f))
    self.assertEqual(expected, result)