from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def get_audio_time_diff(self, audio_time):
    """Query the difference between the provided time and the high
        level `Player`'s master clock.

        The time difference returned is calculated as an average on previous
        audio time differences.

        Return a tuple of the bytes the player is off by, aligned to correspond
        to an integer number of audio frames, as well as bool designating
        whether the difference is extreme. If it is, it should be rectified
        immediately and all previous measurements will have been cleared.

        This method will return ``0, False`` if the difference is not
        significant or ``audio_time`` is ``None``.

        :rtype: int, bool
        """
    required_measurement_count = self.audio_sync_measurements.maxlen
    if audio_time is not None:
        p_time = self.player.time
        audio_time += self.player.last_seek_time
        diff_bytes = self.source.audio_format.timestamp_to_bytes_aligned(audio_time - p_time)
        if abs(diff_bytes) >= self.desync_bytes_critical:
            self.audio_sync_measurements.clear()
            self.audio_sync_cumul_measurements = 0
            return (diff_bytes, True)
        if len(self.audio_sync_measurements) == required_measurement_count:
            self.audio_sync_cumul_measurements -= self.audio_sync_measurements[0]
        self.audio_sync_measurements.append(diff_bytes)
        self.audio_sync_cumul_measurements += diff_bytes
    if len(self.audio_sync_measurements) == required_measurement_count:
        avg_diff = self.source.audio_format.align(self.audio_sync_cumul_measurements // required_measurement_count)
        if abs(avg_diff) > self.desync_bytes_minor:
            return (avg_diff, False)
    return (0, False)