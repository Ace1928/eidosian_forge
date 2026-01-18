import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def auto_wrap_for_ansi(stream: t.TextIO, color: t.Optional[bool]=None) -> t.TextIO:
    """Support ANSI color and style codes on Windows by wrapping a
        stream with colorama.
        """
    try:
        cached = _ansi_stream_wrappers.get(stream)
    except Exception:
        cached = None
    if cached is not None:
        return cached
    import colorama
    strip = should_strip_ansi(stream, color)
    ansi_wrapper = colorama.AnsiToWin32(stream, strip=strip)
    rv = t.cast(t.TextIO, ansi_wrapper.stream)
    _write = rv.write

    def _safe_write(s):
        try:
            return _write(s)
        except BaseException:
            ansi_wrapper.reset_all()
            raise
    rv.write = _safe_write
    try:
        _ansi_stream_wrappers[stream] = rv
    except Exception:
        pass
    return rv