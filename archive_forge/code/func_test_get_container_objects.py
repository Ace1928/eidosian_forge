import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def test_get_container_objects(self):
    with patch('geventhttpclient.HTTPClient.request', lambda *args: Response(content=json.dumps(({'name': 'a'}, {'name': 'b'})))):
        self.assertEqual(len(self.conn.get_container_objects()), 2)