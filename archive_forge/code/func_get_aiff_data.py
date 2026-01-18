import aifc
import audioop
import io
import os
import platform
import stat
import subprocess
import sys
import wave
def get_aiff_data(self, convert_rate=None, convert_width=None):
    """
        Returns a byte string representing the contents of an AIFF-C file containing the audio represented by the ``AudioData`` instance.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        Writing these bytes directly to a file results in a valid `AIFF-C file <https://en.wikipedia.org/wiki/Audio_Interchange_File_Format>`__.
        """
    raw_data = self.get_raw_data(convert_rate, convert_width)
    sample_rate = self.sample_rate if convert_rate is None else convert_rate
    sample_width = self.sample_width if convert_width is None else convert_width
    if hasattr(audioop, 'byteswap'):
        raw_data = audioop.byteswap(raw_data, sample_width)
    else:
        raw_data = raw_data[sample_width - 1::-1] + b''.join((raw_data[i + sample_width:i:-1] for i in range(sample_width - 1, len(raw_data), sample_width)))
    with io.BytesIO() as aiff_file:
        aiff_writer = aifc.open(aiff_file, 'wb')
        try:
            aiff_writer.setframerate(sample_rate)
            aiff_writer.setsampwidth(sample_width)
            aiff_writer.setnchannels(1)
            aiff_writer.writeframes(raw_data)
            aiff_data = aiff_file.getvalue()
        finally:
            aiff_writer.close()
    return aiff_data