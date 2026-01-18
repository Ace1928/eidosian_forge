import sys
import locale
import warnings
def get_stream_enc(stream, default=None):
    """Return the given stream's encoding or a default.

    There are cases where ``sys.std*`` might not actually be a stream, so
    check for the encoding attribute prior to returning it, and return
    a default if it doesn't exist or evaluates as False. ``default``
    is None if not provided.
    """
    if not hasattr(stream, 'encoding') or not stream.encoding:
        return default
    else:
        return stream.encoding