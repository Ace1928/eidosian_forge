import unittest
from os.path import abspath, dirname, join
import errno
import os
def test_json_storage(self):
    from kivy.storage.jsonstore import JsonStore
    from tempfile import mkstemp
    from os import unlink, close
    try:
        tmpfd, tmpfn = mkstemp('.json')
        close(tmpfd)
        self._do_store_test_empty(JsonStore(tmpfn))
        self._do_store_test_filled(JsonStore(tmpfn))
    finally:
        unlink(tmpfn)
    try:
        tmpfd, tmpfn = mkstemp('.json')
        close(tmpfd)
        self._do_store_test_empty(JsonStore(tmpfn, indent=2))
        self._do_store_test_filled(JsonStore(tmpfn, indent=2))
    finally:
        unlink(tmpfn)
    try:
        tmpfd, tmpfn = mkstemp('.json')
        close(tmpfd)
        self._do_store_test_empty(JsonStore(tmpfn, sort_keys=True))
        self._do_store_test_filled(JsonStore(tmpfn, sort_keys=True))
    finally:
        unlink(tmpfn)