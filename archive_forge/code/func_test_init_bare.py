import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_init_bare(self):
    swift.SwiftRepo.init_bare(self.scon, self.conf)
    self.assertTrue(self.scon.test_root_exists())
    obj = self.scon.get_container_objects()
    filtered = [o for o in obj if o['name'] == 'info/refs' or o['name'] == 'objects/pack']
    self.assertEqual(len(filtered), 2)