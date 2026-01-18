import re
from hacking import core
@core.flake8ext
def check_python3_no_iterkeys(logical_line):
    msg = 'HE303: Use dict.keys() instead of dict.iterkeys().'
    if re.search('.*\\.iterkeys\\(\\)', logical_line):
        yield (0, msg)