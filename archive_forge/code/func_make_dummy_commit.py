import copy
import http.server
import os
import select
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from contextlib import suppress
from io import BytesIO
from urllib.parse import unquote
from dulwich import client, file, index, objects, protocol, repo
from dulwich.tests import SkipTest, expectedFailure
from .utils import (
def make_dummy_commit(self, dest):
    b = objects.Blob.from_string(b'hi')
    dest.object_store.add_object(b)
    t = index.commit_tree(dest.object_store, [(b'hi', b.id, 33188)])
    c = objects.Commit()
    c.author = c.committer = b'Foo Bar <foo@example.com>'
    c.author_time = c.commit_time = 0
    c.author_timezone = c.commit_timezone = 0
    c.message = b'hi'
    c.tree = t
    dest.object_store.add_object(c)
    return c.id