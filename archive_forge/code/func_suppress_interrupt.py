import signal
from .utils import TimeoutException, BaseTimeout, base_timeoutable
def suppress_interrupt(self):
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)