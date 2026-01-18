import re
from hacking import core
from neutron_lib.hacking import translation_checks
@core.flake8ext
def check_neutron_namespace_imports(logical_line):
    """N530 - Direct neutron imports not allowed.

    :param logical_line: The logical line to check.
    :returns: None if the logical line passes the check, otherwise a tuple
        is yielded that contains the offending index in logical line and a
        message describe the check validation failure.
    """
    x = _check_namespace_imports('N530', 'neutron', 'neutron_lib.', logical_line, message_override='direct neutron imports not allowed')
    if x is not None:
        yield x