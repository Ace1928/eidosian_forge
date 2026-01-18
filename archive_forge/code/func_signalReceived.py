import weakref
from time import perf_counter
from .functions import SignalBlock
from .Qt import QtCore
from .ThreadsafeTimer import ThreadsafeTimer
def signalReceived(self, *args):
    """Received signal. Cancel previous timer and store args to be
        forwarded later."""
    if self.blockSignal:
        return
    self.args = args
    if self.rateLimit == 0:
        self.timer.stop()
        self.timer.start(int(self.delay * 1000) + 1)
    else:
        now = perf_counter()
        if self.lastFlushTime is None:
            leakTime = 0
        else:
            lastFlush = self.lastFlushTime
            leakTime = max(0, lastFlush + 1.0 / self.rateLimit - now)
        self.timer.stop()
        self.timer.start(int(min(leakTime, self.delay) * 1000) + 1)