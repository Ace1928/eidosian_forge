import inspect
import logging
import urllib.parse
def make_range_string(start=None, stop=None):
    """Create a byte range specifier in accordance with RFC-2616.

    Parameters
    ----------
    start: int, optional
        The start of the byte range.  If unspecified, stop indicated offset from EOF.

    stop: int, optional
        The end of the byte range.  If unspecified, indicates EOF.

    Returns
    -------
    str
        A byte range specifier.

    """
    if start is None and stop is None:
        raise ValueError('make_range_string requires either a stop or start value')
    start_str = '' if start is None else str(start)
    stop_str = '' if stop is None else str(stop)
    return 'bytes=%s-%s' % (start_str, stop_str)