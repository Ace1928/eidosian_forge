import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def compare_expected(self, expected, comparison=operator.eq, differ=None, not_found=None, include_success=False):
    """
        Compares a dictionary of ``path: content`` to the
        found files.  Comparison is done by equality, or the
        ``comparison(actual_content, expected_content)`` function given.

        Returns dictionary of differences, keyed by path.  Each
        difference is either noted, or the output of
        ``differ(actual_content, expected_content)`` is given.

        If a file does not exist and ``not_found`` is given, then
        ``not_found(path)`` is put in.
        """
    result = {}
    for path in expected:
        orig_path = path
        path = path.strip('/')
        if path not in self.data:
            if not_found:
                msg = not_found(path)
            else:
                msg = 'not found'
            result[path] = msg
            continue
        expected_content = expected[orig_path]
        file = self.data[path]
        actual_content = file.bytes
        if not comparison(actual_content, expected_content):
            if differ:
                msg = differ(actual_content, expected_content)
            elif len(actual_content) < len(expected_content):
                msg = 'differ (%i bytes smaller)' % (len(expected_content) - len(actual_content))
            elif len(actual_content) > len(expected_content):
                msg = 'differ (%i bytes larger)' % (len(actual_content) - len(expected_content))
            else:
                msg = 'diff (same size)'
            result[path] = msg
        elif include_success:
            result[path] = 'same!'
    return result