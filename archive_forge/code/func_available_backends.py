from . import ffdec
from .exceptions import DecodeError, NoBackendError
from .version import version as __version__  # noqa
from .base import AudioFile  # noqa
def available_backends(flush_cache=False):
    """Returns a list of backends that are available on this system.

    The list of backends is cached after the first call.
    If the parameter `flush_cache` is set to `True`, then the cache
    will be flushed and the backend list will be reconstructed.
    """
    if BACKENDS and (not flush_cache):
        return BACKENDS
    from . import rawread
    result = [rawread.RawAudioFile]
    if _ca_available():
        from . import macca
        result.append(macca.ExtAudioFile)
    if _gst_available():
        from . import gstdec
        result.append(gstdec.GstAudioFile)
    if _mad_available():
        from . import maddec
        result.append(maddec.MadAudioFile)
    if ffdec.available():
        result.append(ffdec.FFmpegAudioFile)
    BACKENDS[:] = result
    return BACKENDS