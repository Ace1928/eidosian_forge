import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
def test_get_tagged(self):
    refs = {b'refs/tags/tag1': ONE, b'refs/tags/tag2': TWO, b'refs/heads/master': FOUR}
    self._repo.object_store.add_object(make_commit(id=FOUR))
    for name, sha in refs.items():
        self._repo.refs[name] = sha
    peeled = {b'refs/tags/tag1': b'1234' * 10, b'refs/tags/tag2': b'5678' * 10}
    self._repo.refs._peeled_refs = peeled
    self._repo.refs.add_packed_refs(refs)
    caps = [*list(self._handler.required_capabilities()), b'include-tag']
    self._handler.set_client_capabilities(caps)
    self.assertEqual({b'1234' * 10: ONE, b'5678' * 10: TWO}, self._handler.get_tagged(refs, repo=self._repo))
    caps = self._handler.required_capabilities()
    self._handler.set_client_capabilities(caps)
    self.assertEqual({}, self._handler.get_tagged(refs, repo=self._repo))