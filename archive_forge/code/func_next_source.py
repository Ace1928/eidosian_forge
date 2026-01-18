from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def next_source(self) -> None:
    """Move immediately to the next source in the current playlist.

        If the playlist is empty, discard it and check if another playlist
        is queued. There may be a gap in playback while the audio buffer
        is refilled.
        """
    was_playing = self._playing
    self.pause()
    self._timer.reset()
    self.last_seek_time = 0.0
    if self._source:
        self.seek(0.0)
        self.source.is_player_source = False
    playlists = self._playlists
    if not playlists:
        return
    try:
        new_source = next(playlists[0])
    except StopIteration:
        playlists.popleft()
        if not playlists:
            new_source = None
        else:
            new_source = next(playlists[0])
    if new_source is None:
        self._source = None
        self.delete()
        self.dispatch_event('on_player_eos')
    else:
        old_source = self._source
        self._set_source(new_source)
        if self._audio_player is not None:
            if self._source.audio_format == old_source.audio_format:
                self._audio_player.set_source(self._source)
            else:
                self._audio_player.delete()
                self._audio_player = None
        if self._source.video_format != old_source.video_format:
            self._texture = None
            pyglet.clock.unschedule(self.update_texture)
        del old_source
        self._set_playing(was_playing)
        self.dispatch_event('on_player_next_source')