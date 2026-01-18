import re
from hacking import core
@core.flake8ext
def check_no_logging_imports(logical_line):
    """O348 - Usage of Python logging module not allowed.

    :param logical_line: The logical line to check.
    :returns: None if the logical line passes the check, otherwise a tuple
              is yielded that contains the offending index in logical line
              and a message describe the check validation failure.
    """
    if no_logging_re.match(logical_line):
        msg = 'O348 Usage of Python logging module not allowed, use oslo_log'
        yield (logical_line.index('logging'), msg)