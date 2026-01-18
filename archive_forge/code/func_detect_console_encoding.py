from __future__ import annotations
import locale
import sys
from pandas._config import config as cf
def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """
    global _initial_defencoding
    encoding = None
    try:
        encoding = sys.stdout.encoding or sys.stdin.encoding
    except (AttributeError, OSError):
        pass
    if not encoding or 'ascii' in encoding.lower():
        try:
            encoding = locale.getpreferredencoding()
        except locale.Error:
            pass
    if not encoding or 'ascii' in encoding.lower():
        encoding = sys.getdefaultencoding()
    if not _initial_defencoding:
        _initial_defencoding = sys.getdefaultencoding()
    return encoding