import codecs
import gzip
import os
import six.moves.urllib.request as urllib_request
import tempfile
import unittest
from apitools.gen import util
from mock import patch
def testGZippedContent(self):
    data = u'¿Hola qué tal?'
    compressed_data = _Gzip(data.encode('utf-8'))
    with patch.object(urllib_request, 'urlopen', return_value=MockRequestResponse(compressed_data, 'gzip')):
        self.assertEqual(data, util._GetURLContent('unused_url_parameter').decode('utf-8'))