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
def test_image_show_invalid_endpoint(self):
    self.assertRaises(exc.CommunicationError, self.run_command, 'image-show --human-readable <IMAGE_ID>')