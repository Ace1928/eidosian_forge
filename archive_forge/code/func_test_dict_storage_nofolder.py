import unittest
from os.path import abspath, dirname, join
import errno
import os
def test_dict_storage_nofolder(self):
    from kivy.storage.dictstore import DictStore
    self._do_store_test_nofolder(DictStore)