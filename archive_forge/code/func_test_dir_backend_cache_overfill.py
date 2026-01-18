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
def test_dir_backend_cache_overfill(self):
    if self.max_cache_size is not None:
        books_ids_made = []
        with contextlib.closing(self._get_connection()) as conn:
            for i in range(0, int(1.5 * self.max_cache_size)):
                lb_name = 'book-%s' % i
                lb_id = uuidutils.generate_uuid()
                lb = models.LogBook(name=lb_name, uuid=lb_id)
                self.assertRaises(exc.NotFound, conn.get_logbook, lb_id)
                conn.save_logbook(lb)
                books_ids_made.append(lb_id)
                self.assertLessEqual(self.backend.file_cache.currsize, self.max_cache_size)
        with contextlib.closing(self._get_connection()) as conn:
            for lb_id in books_ids_made:
                lb = conn.get_logbook(lb_id)
                self.assertIsNotNone(lb)