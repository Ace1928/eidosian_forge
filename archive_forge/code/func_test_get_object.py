import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_get_object(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(content=b'content')):
        self.assertEqual(self.conn.get_object('a').read(), b'content')
    with patch('geventhttpclient.HTTPClient.request', lambda *args, **kwargs: Response(content=b'content')):
        self.assertEqual(self.conn.get_object('a', range='0-6'), b'content')