import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
def test_delete_absent_key(self):
    cache = self.make_file_cache()
    cache.delete('nonexistent')
    self.assertIs(None, cache.get('nonexistent'))