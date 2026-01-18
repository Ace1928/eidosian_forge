import io
import sys
from unittest import mock
from oslo_utils import encodeutils
from requests import Response
import testtools
from glanceclient.common import utils
def test_iterable_closes(self):

    def _iterate(i):
        for chunk in i:
            raise IOError()
    data = io.StringIO('somestring')
    data.close = mock.Mock()
    i = utils.IterableWithLength(data, 10)
    self.assertRaises(IOError, _iterate, i)
    data.close.assert_called_with()