import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def get_size_from_qemu_img(filename):
    output = subprocess.check_output('qemu-img info "%s"' % filename, shell=True)
    for line in output.split(b'\n'):
        m = re.search(b'^virtual size: .* .([0-9]+) bytes', line.strip())
        if m:
            return int(m.group(1))
    raise Exception('Could not find virtual size with qemu-img')