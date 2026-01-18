from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
def on_player_next_source(self):
    """The player starts to play the next queued source in the playlist.

        This is a useful event for adjusting the window size to the new
        source :class:`VideoFormat` for example.

        :event:
        """
    pass