import aifc
import audioop
import struct
import sunau
import wave
from .exceptions import DecodeError
from .base import AudioFile
@property
def samplerate(self):
    """Sample rate in Hz."""
    return self._file.getframerate()