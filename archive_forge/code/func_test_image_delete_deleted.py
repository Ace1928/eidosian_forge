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
def test_image_delete_deleted(self):
    self.assertRaises(exc.CommandError, v1shell.do_image_delete, self.gc, argparse.Namespace(images=['70aa106f-3750-4d7c-a5ce-0a535ac08d0a']))