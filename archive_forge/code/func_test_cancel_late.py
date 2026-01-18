import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_cancel_late(self):

    def foo(*args, **kwargs):
        time.sleep(0.3)
    self.tg.add_thread(foo, 'arg', kwarg='kwarg')
    time.sleep(0)
    self.tg.cancel()
    self.assertEqual(1, len(self.tg.threads))