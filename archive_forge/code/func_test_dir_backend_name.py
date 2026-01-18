import contextlib
import os
import shutil
import tempfile
from oslo_utils import uuidutils
import testscenarios
from taskflow import exceptions as exc
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_dir
from taskflow.persistence import models
from taskflow import test
from taskflow.tests.unit.persistence import base
def test_dir_backend_name(self):
    self._check_backend(dict(connection='dir', path=self.path))