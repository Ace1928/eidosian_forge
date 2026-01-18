import time
from eventlet import event
from oslotest import base as test_base
from oslo_service import threadgroup
def stop_self(*args, **kwargs):
    if args[0] == 1:
        time.sleep(1)
        self.tg.stop()
        stop_event.send('stop_event')
    quit_event.wait()