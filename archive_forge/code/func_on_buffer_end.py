from collections import deque
import math
import threading
from typing import Deque, Tuple, TYPE_CHECKING
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.media.drivers.listener import AbstractListener
from pyglet.util import debug_print
from . import interface
def on_buffer_end(self, buffer_context_ptr: int) -> None:
    with self._audio_data_lock:
        assert self._audio_data_in_use
        self._audio_data_in_use.popleft()
        if self._audio_data_in_use:
            assert _debug(f'Buffer ended, others remain: len(self._audio_data_in_use)={len(self._audio_data_in_use)!r}')
            return
        assert self._xa2_source_voice.buffers_queued == 0
        if self._pyglet_source_exhausted:
            assert _debug('Last buffer ended normally, dispatching eos')
            MediaEvent('on_eos').sync_dispatch_to_player(self.player)
        else:
            assert _debug('Last buffer ended normally, source is lagging behind')