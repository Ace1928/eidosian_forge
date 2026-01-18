import abc
import contextlib
import os
import random
import tempfile
import testtools
import sqlalchemy as sa
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_sqlalchemy
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_entrypoint(self):
    with contextlib.closing(backends.fetch(self.db_conf)) as backend:
        with contextlib.closing(backend.get_connection()):
            pass