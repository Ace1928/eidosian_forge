import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@ddt.data('loginsight://username@host', 'loginsight://username:p@ssword@host', 'loginsight://us:rname:password@host')
def test_init_with_invalid_connection_string(self, conn_str):
    self.assertRaises(ValueError, loginsight.LogInsightDriver, conn_str)