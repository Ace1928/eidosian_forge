import argparse
from copy import deepcopy
import io
import json
import os
from unittest import mock
import sys
import tempfile
import testtools
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell
from glanceclient.v2 import shell as test_shell  # noqa
def schema_args(schema_getter, omit=None):
    global original_schema_args
    my_schema_getter = lambda: {'properties': {'container_format': {'enum': [None, 'ami', 'ari', 'aki', 'bare', 'ovf', 'ova', 'docker'], 'type': 'string', 'description': 'Format of the container'}, 'disk_format': {'enum': [None, 'ami', 'ari', 'aki', 'vhd', 'vhdx', 'vmdk', 'raw', 'qcow2', 'vdi', 'iso', 'ploop'], 'type': 'string', 'description': 'Format of the disk'}, 'location': {'type': 'string'}, 'locations': {'type': 'string'}, 'copy_from': {'type': 'string'}}}
    return original_schema_args(my_schema_getter, omit)