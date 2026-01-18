import re
from hacking import core
import pycodestyle
@core.flake8ext
def check_no_log_warn(logical_line):
    """Disallow 'LOG.warn('

    D710
    """
    if logical_line.startswith('LOG.warn('):
        yield (0, 'D710:Use LOG.warning() rather than LOG.warn()')