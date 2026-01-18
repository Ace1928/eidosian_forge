import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def assertRegexMatch(self, actual_str, regexes, message=None):
    """Asserts that at least one regex in regexes matches str.

    If possible you should use assertRegexpMatches, which is a simpler
    version of this method. assertRegexpMatches takes a single regular
    expression (a string or re compiled object) instead of a list.

    Notes:
    1. This function uses substring matching, i.e. the matching
       succeeds if *any* substring of the error message matches *any*
       regex in the list.  This is more convenient for the user than
       full-string matching.

    2. If regexes is the empty list, the matching will always fail.

    3. Use regexes=[''] for a regex that will always pass.

    4. '.' matches any single character *except* the newline.  To
       match any character, use '(.|
)'.

    5. '^' matches the beginning of each line, not just the beginning
       of the string.  Similarly, '$' matches the end of each line.

    6. An exception will be thrown if regexes contains an invalid
       regex.

    Args:
      actual_str:  The string we try to match with the items in regexes.
      regexes:  The regular expressions we want to match against str.
        See "Notes" above for detailed notes on how this is interpreted.
      message:  The message to be printed if the test fails.
    """
    if isinstance(regexes, basestring):
        self.fail('regexes is a string; it needs to be a list of strings.')
    if not regexes:
        self.fail('No regexes specified.')
    regex = '(?:%s)' % ')|(?:'.join(regexes)
    if not re.search(regex, actual_str, re.MULTILINE):
        self.fail(message or 'String "%s" does not contain any of these regexes: %s.' % (actual_str, regexes))