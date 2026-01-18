import unittest
from os.path import abspath, dirname, join
import errno
import os
def test_json_storage_nofolder(self):
    from kivy.storage.jsonstore import JsonStore
    self._do_store_test_nofolder(JsonStore)