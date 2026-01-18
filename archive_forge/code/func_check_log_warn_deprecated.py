import re
from hacking import core
@core.flake8ext
def check_log_warn_deprecated(logical_line, filename):
    """N532 - Use LOG.warning due to compatibility with py3.

    :param logical_line: The logical line to check.
    :param filename: The file name where the logical line exists.
    :returns: None if the logical line passes the check, otherwise a tuple
        is yielded that contains the offending index in logical line and a
        message describe the check validation failure.
    """
    msg = 'N532: Use LOG.warning due to compatibility with py3'
    if _log_warn.match(logical_line):
        yield (0, msg)