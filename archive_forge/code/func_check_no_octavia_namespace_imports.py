import re
from hacking import core
@core.flake8ext
def check_no_octavia_namespace_imports(logical_line):
    """O501 - Direct octavia imports not allowed.

    :param logical_line: The logical line to check.
    :returns: None if the logical line passes the check, otherwise a tuple
        is yielded that contains the offending index in logical line and a
        message describe the check validation failure.
    """
    x = _check_namespace_imports('O501', 'octavia', 'octavia_lib.', logical_line, message_override='O501 Direct octavia imports not allowed')
    if x is not None:
        yield x