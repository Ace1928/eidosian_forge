import atexit
import os
import signal
import sys
import ovs.vlog
def signal_alarm(timeout):
    if not timeout:
        env_timeout = os.environ.get('OVS_CTL_TIMEOUT')
        if env_timeout:
            timeout = int(env_timeout)
    if not timeout:
        return
    if sys.platform == 'win32':
        import time
        import threading

        class Alarm(threading.Thread):

            def __init__(self, timeout):
                super(Alarm, self).__init__()
                self.timeout = timeout
                self.setDaemon(True)

            def run(self):
                time.sleep(self.timeout)
                os._exit(1)
        alarm = Alarm(timeout)
        alarm.start()
    else:
        signal.alarm(timeout)