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
def test_image_update_opened_stdin(self):
    """Test image update with an opened stdin.

        Supply glanceclient with a stdin, and perform an image
        update to an active image. Glanceclient should not allow it.
        """
    self.assertRaises(SystemExit, v1shell.do_image_update, self.gc, argparse.Namespace(image='96d2c7e1-de4e-4612-8aa2-ba26610c804e', property={}))