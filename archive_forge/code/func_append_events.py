from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def append_events(self, start_index, events):
    """Append the given :class:`MediaEvent`s to the events deque using
        the current source's audio format and the supplied ``start_index``
        to convert their timestamps to dispatch indices.

        The high level player's ``last_seek_time`` will be subtracted from
        each event's timestamp.
        """
    bps = self.source.audio_format.bytes_per_second
    lst = self.player.last_seek_time
    for event in events:
        event_cursor = start_index + max(0.0, event.timestamp - lst) * bps
        assert _debug(f'AbstractAudioPlayer: Adding event {event} at {event_cursor}')
        self._events.append((event_cursor, event))