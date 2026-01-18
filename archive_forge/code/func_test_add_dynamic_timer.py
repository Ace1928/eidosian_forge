import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def test_add_dynamic_timer(self):

    def foo(*args, **kwargs):
        pass
    initial_delay = 1
    periodic_interval_max = 2
    self.tg.add_dynamic_timer(foo, initial_delay, periodic_interval_max, 'arg', kwarg='kwarg')
    self.assertEqual(1, len(self.tg.timers))
    timer = self.tg.timers[0]
    self.assertTrue(timer._running)
    self.assertEqual(('arg',), timer.args)
    self.assertEqual({'kwarg': 'kwarg'}, timer.kw)