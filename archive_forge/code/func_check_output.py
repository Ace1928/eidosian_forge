import doctest
import re
import sys
def check_output(self, want, got, optionflags):
    if sys.version_info < (3, 0):
        got = re.sub("u'(.*?)'", "'\\1'", got)
    return doctest.OutputChecker.check_output(self, want, got, optionflags)