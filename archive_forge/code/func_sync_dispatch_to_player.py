from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def sync_dispatch_to_player(self, player):
    pyglet.app.platform_event_loop.post_event(player, self.event, *self.args)