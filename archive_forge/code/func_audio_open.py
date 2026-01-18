from . import ffdec
from .exceptions import DecodeError, NoBackendError
from .version import version as __version__  # noqa
from .base import AudioFile  # noqa
def audio_open(path, backends=None):
    """Open an audio file using a library that is available on this
    system.

    The optional `backends` parameter can be a list of audio file
    classes to try opening the file with. If it is not provided,
    `audio_open` tries all available backends. If you call this function
    many times, you can avoid the cost of checking for available
    backends every time by calling `available_backends` once and passing
    the result to each `audio_open` call.

    If all backends fail to read the file, a NoBackendError exception is
    raised.
    """
    if backends is None:
        backends = available_backends()
    for BackendClass in backends:
        try:
            return BackendClass(path)
        except DecodeError:
            pass
    raise NoBackendError()