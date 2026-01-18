import re
from hacking import core
@core.flake8ext
def check_oslo_namespace_imports(logical_line, blank_before, filename):
    oslo_namespace_imports = re.compile('(((from)|(import))\\s+oslo\\.)|(from\\s+oslo\\s+import\\s+)')
    if re.match(oslo_namespace_imports, logical_line):
        msg = "K333: '%s' must be used instead of '%s'." % (logical_line.replace('oslo.', 'oslo_'), logical_line)
        yield (0, msg)