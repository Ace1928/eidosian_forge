import re
from hacking import core
@core.flake8ext
def no_log_warn(logical_line):
    """Disallow 'LOG.warn('

    Use LOG.warning() instead of Deprecated LOG.warn().
    https://docs.python.org/3/library/logging.html#logging.warning
    """
    msg = 'G330: LOG.warn is deprecated, please use LOG.warning!'
    if 'LOG.warn(' in logical_line:
        yield (0, msg)