import argparse
import collections
import contextlib
import io
import re
import tokenize
from typing import TextIO, Tuple
import untokenize  # type: ignore
import docformatter.encode as _encode
import docformatter.strings as _strings
import docformatter.syntax as _syntax
import docformatter.util as _util
def do_format_files(self):
    """Format multiple files.

        Return
        ------
        code: int
            One of the FormatResult codes.
        """
    outcomes = collections.Counter()
    for filename in _util.find_py_files(set(self.args.files), self.args.recursive, self.args.exclude):
        try:
            result = self._do_format_file(filename)
            outcomes[result] += 1
        except OSError as exception:
            outcomes[FormatResult.error] += 1
            print(unicode(exception), file=self.stderror)
    return_codes = [FormatResult.error, FormatResult.check_failed, FormatResult.ok]
    for code in return_codes:
        if outcomes[code]:
            return code