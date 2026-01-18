from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib import cloud_api
from gslib import gcs_json_api
from gslib import context_config
from gslib.tests import testcase
from gslib.tests.testcase import base
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from six import add_move, MovedModule
from six.moves import mock
def testSetsDefaultHost(self):
    with SetBotoConfigForTest([('Credentials', 'gs_json_host', None), ('Credentials', 'gs_host', None)]):
        client = gcs_json_api.GcsJsonApi(None, None, None, None)
        self.assertEqual(client.host_base, gcs_json_api.DEFAULT_HOST)