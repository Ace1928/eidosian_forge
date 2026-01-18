import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_cancel_early(self):

    def foo(*args, **kwargs):
        time.sleep(1)
    self.tg.add_thread(foo, 'arg', kwarg='kwarg')
    self.tg.cancel()
    self.assertEqual(0, len(self.tg.threads))