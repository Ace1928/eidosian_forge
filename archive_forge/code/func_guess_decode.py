import re
import sys
def guess_decode(text):
    """Decode *text* with guessed encoding.

    First try UTF-8; this should fail for non-UTF-8 encodings.
    Then try the preferred locale encoding.
    Fall back to latin-1, which always works.
    """
    try:
        text = text.decode('utf-8')
        return (text, 'utf-8')
    except UnicodeDecodeError:
        try:
            import locale
            prefencoding = locale.getpreferredencoding()
            text = text.decode()
            return (text, prefencoding)
        except (UnicodeDecodeError, LookupError):
            text = text.decode('latin1')
            return (text, 'latin1')