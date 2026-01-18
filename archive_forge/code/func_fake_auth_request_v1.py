import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def fake_auth_request_v1(*args, **kwargs):
    ret = Response({'X-Storage-Url': 'http://127.0.0.1:8080/v1.0/AUTH_fakeuser', 'X-Auth-Token': '12' * 10}, 200)
    return ret