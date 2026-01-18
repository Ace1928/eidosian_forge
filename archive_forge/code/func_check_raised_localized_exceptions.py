import re
from hacking import core
@core.flake8ext
def check_raised_localized_exceptions(logical_line, filename):
    """N534 - Untranslated exception message.

    :param logical_line: The logical line to check.
    :param filename: The file name where the logical line exists.
    :returns: None if the logical line passes the check, otherwise a tuple
        is yielded that contains the offending index in logical line and a
        message describe the check validation failure.
    """
    if _translation_checks_not_enforced(filename):
        return
    logical_line = logical_line.strip()
    raised_search = re.compile('raise (?:\\w*)\\((.*)\\)').match(logical_line)
    if raised_search:
        exception_msg = raised_search.groups()[0]
        if exception_msg.startswith('"') or exception_msg.startswith("'"):
            msg = 'N534: Untranslated exception message.'
            yield (logical_line.index(exception_msg), msg)