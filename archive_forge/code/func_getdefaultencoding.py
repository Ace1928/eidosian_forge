import sys
import locale
import warnings
def getdefaultencoding(prefer_stream=True):
    """Return IPython's guess for the default encoding for bytes as text.

    If prefer_stream is True (default), asks for stdin.encoding first,
    to match the calling Terminal, but that is often None for subprocesses.

    Then fall back on locale.getpreferredencoding(),
    which should be a sensible platform default (that respects LANG environment),
    and finally to sys.getdefaultencoding() which is the most conservative option,
    and usually UTF8 as of Python 3.
    """
    enc = None
    if prefer_stream:
        enc = get_stream_enc(sys.stdin)
    if not enc or enc == 'ascii':
        try:
            enc = locale.getpreferredencoding()
        except Exception:
            pass
    enc = enc or sys.getdefaultencoding()
    if enc == 'cp0':
        warnings.warn('Invalid code page cp0 detected - using cp1252 instead.If cp1252 is incorrect please ensure a valid code page is defined for the process.', RuntimeWarning)
        return 'cp1252'
    return enc