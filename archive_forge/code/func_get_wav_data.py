import aifc
import audioop
import io
import os
import platform
import stat
import subprocess
import sys
import wave
def get_wav_data(self, convert_rate=None, convert_width=None):
    """
        Returns a byte string representing the contents of a WAV file containing the audio represented by the ``AudioData`` instance.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        Writing these bytes directly to a file results in a valid `WAV file <https://en.wikipedia.org/wiki/WAV>`__.
        """
    raw_data = self.get_raw_data(convert_rate, convert_width)
    sample_rate = self.sample_rate if convert_rate is None else convert_rate
    sample_width = self.sample_width if convert_width is None else convert_width
    with io.BytesIO() as wav_file:
        wav_writer = wave.open(wav_file, 'wb')
        try:
            wav_writer.setframerate(sample_rate)
            wav_writer.setsampwidth(sample_width)
            wav_writer.setnchannels(1)
            wav_writer.writeframes(raw_data)
            wav_data = wav_file.getvalue()
        finally:
            wav_writer.close()
    return wav_data