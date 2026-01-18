from collections import deque
import math
import threading
from typing import Deque, Tuple, TYPE_CHECKING
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.media.drivers.listener import AbstractListener
from pyglet.util import debug_print
from . import interface
def on_driver_reset(self) -> None:
    self._xa2_source_voice = self.driver._xa2_driver.get_source_voice(self.source.audio_format, self)
    for audio_data in self._audio_data_in_use:
        xa2_buffer = interface.create_xa2_buffer(audio_data)
        self._xa2_source_voice.submit_buffer(xa2_buffer)