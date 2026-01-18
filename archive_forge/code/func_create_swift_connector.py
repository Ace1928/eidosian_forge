import json
import posixpath
from io import BytesIO, StringIO
from time import time
from unittest import skipIf
from dulwich.tests import TestCase
from ..objects import Blob, Commit, Tag, Tree, parse_timezone
from ..tests.test_object_store import ObjectStoreTests
def create_swift_connector(store={}):
    return lambda root, conf: FakeSwiftConnector(root, conf=conf, store=store)