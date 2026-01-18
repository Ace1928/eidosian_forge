import argparse
import io
import json
import os
from unittest import mock
import subprocess
import tempfile
import testtools
from glanceclient import exc
from glanceclient import shell
import glanceclient.v1.client as client
import glanceclient.v1.images
import glanceclient.v1.shell as v1shell
from glanceclient.tests import utils
def test_image_update_data_is_read_from_file(self):
    """Ensure that data is read from a file."""
    try:
        f = open(tempfile.mktemp(), 'w+')
        f.write('Some Data')
        f.flush()
        f.seek(0)
        os.dup2(f.fileno(), 0)
        self._do_update('44d2c7e1-de4e-4612-8aa2-ba26610c444f')
        self.assertIn('data', self.collected_args[1])
        self.assertIsInstance(self.collected_args[1]['data'], io.IOBase)
        self.assertEqual(b'Some Data', self.collected_args[1]['data'].read())
    finally:
        try:
            f.close()
            os.remove(f.name)
        except Exception:
            pass