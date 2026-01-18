import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def check_context_error(self, message=None):
    """Check whether there is an OpenAL error and raise exception if present."""
    error_code = alc.alcGetError(self._al_device)
    if error_code != 0:
        error_string = alc.alcGetString(self._al_device, error_code)
        error_string = ctypes.cast(error_string, ctypes.c_char_p)
        raise OpenALException(message=message, error_code=error_code, error_string=str(error_string.value))