import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_add_and_remove_dynamic_timer(self):

    def foo(*args, **kwargs):
        pass
    initial_delay = 1
    periodic_interval_max = 2
    timer = self.tg.add_dynamic_timer(foo, initial_delay, periodic_interval_max)
    self.assertEqual(1, len(self.tg.timers))
    self.assertTrue(timer._running)
    timer.stop()
    self.assertEqual(1, len(self.tg.timers))
    self.tg.timer_done(timer)
    self.assertEqual(0, len(self.tg.timers))