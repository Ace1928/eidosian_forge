import os, re
from distutils.core import Command
from distutils.errors import DistutilsExecError
from distutils.sysconfig import customize_compiler
from distutils import log
def search_cpp(self, pattern, body=None, headers=None, include_dirs=None, lang='c'):
    """Construct a source file (just like 'try_cpp()'), run it through
        the preprocessor, and return true if any line of the output matches
        'pattern'.  'pattern' should either be a compiled regex object or a
        string containing a regex.  If both 'body' and 'headers' are None,
        preprocesses an empty file -- which can be useful to determine the
        symbols the preprocessor and compiler set by default.
        """
    self._check_compiler()
    src, out = self._preprocess(body, headers, include_dirs, lang)
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    with open(out) as file:
        match = False
        while True:
            line = file.readline()
            if line == '':
                break
            if pattern.search(line):
                match = True
                break
    self._clean()
    return match