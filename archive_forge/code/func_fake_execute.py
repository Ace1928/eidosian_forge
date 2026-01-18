import io
import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
from glance.async_.flows import convert
from glance.async_ import taskflow_executor
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def fake_execute(*args, **kwargs):
    if 'info' in args:
        assert os.path.exists(args[3].split('file://')[-1])
        return (json.dumps({'virtual-size': 10737418240, 'filename': '/tmp/image.qcow2', 'cluster-size': 65536, 'format': 'qcow2', 'actual-size': 373030912, 'format-specific': {'type': 'qcow2', 'data': {'compat': '0.10'}}, 'dirty-flag': False}), None)
    open('%s.converted' % image_path, 'a').close()
    return ('', None)