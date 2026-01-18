from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def create_audio_player(self, source: 'Source', player: 'Player') -> 'OpenALAudioPlayer':
    assert self.device is not None, 'Device was closed'
    return OpenALAudioPlayer(self, source, player)