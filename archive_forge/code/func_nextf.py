from reportlab.rl_config import register_reset
def nextf(self, counter=None):
    """Retrieves the numeric value for the given counter, then
        increments it by one.  New counters start at one."""
    if not counter:
        counter = self._defaultCounter
    return self._getCounter(counter).nextf()