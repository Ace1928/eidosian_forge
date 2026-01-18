from unittest import mock
import futurist
import glance_store
from oslo_config import cfg
from taskflow import engines
import glance.async_
from glance.async_ import taskflow_executor
from glance.common.scripts.image_import import main as image_import
from glance import domain
import glance.tests.utils as test_utils
def test_fetch_an_executor_parallel(self):
    self.config(engine_mode='parallel', group='taskflow_executor')
    pool = self.executor._fetch_an_executor()
    self.assertIsInstance(pool, futurist.GreenThreadPoolExecutor)