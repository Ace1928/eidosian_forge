import StringIO
import time
def overhead(self, now=None):
    """Calculate the overhead.

    Args:
      now: (optional) time to use as the current time.

    Returns:
      The overhead, that is, time spent in total but not in any sub timer.  This
      may be negative if time was counted in two sub timers.  Avoid this by
      always using stop_others.
    """
    total = self.timervalue('total', now)
    if total == 0.0:
        return 0.0
    all_timers = sum(self.accum.itervalues())
    return total - (all_timers - total)