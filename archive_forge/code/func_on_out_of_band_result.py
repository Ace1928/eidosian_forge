import time
import kombu
from kombu.common import maybe_declare
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from celery import states
from celery._state import current_task, task_join_will_block
from . import base
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
def on_out_of_band_result(self, task_id, message):
    if self.result_consumer:
        self.result_consumer.on_out_of_band_result(message)
    self._out_of_band[task_id] = message