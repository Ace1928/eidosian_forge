import io
import json
import os
from unittest import mock
import urllib
import glance_store
from oslo_concurrency import processutils as putils
from oslo_config import cfg
from taskflow import task
from taskflow.types import failure
import glance.async_.flows.base_import as import_flow
from glance.async_ import taskflow_executor
from glance.async_ import utils as async_utils
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import utils
from glance import context
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def test_delete_from_fs(self):
    delete_fs = import_flow._DeleteFromFS(self.task.task_id, self.task_type)
    data = [b'test']
    store = glance_store.get_store_from_scheme('file')
    path = glance_store.store_add_to_backend(mock.sentinel.image_id, data, mock.sentinel.image_size, store, context=None)[0]
    path_wo_scheme = path.split('file://')[1]
    self.assertTrue(os.path.exists(path_wo_scheme))
    delete_fs.execute(path)
    self.assertFalse(os.path.exists(path_wo_scheme))