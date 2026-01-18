from collections import deque
import time
from typing import Iterable, Optional, Union
import pyglet
from pyglet.gl import GL_TEXTURE_2D
from pyglet.media import buffered_logger as bl
from pyglet.media.drivers import get_audio_driver
from pyglet.media.codecs.base import PreciseStreamingSource, Source, SourceGroup
@property
def playing(self) -> bool:
    """
        bool: Read-only. Determine if the player state is playing.

        The *playing* property is irrespective of whether or not there is
        actually a source to play. If *playing* is ``True`` and a source is
        queued, it will begin to play immediately. If *playing* is ``False``,
        it is implied that the player is paused. There is no other possible
        state.
        """
    return self._playing