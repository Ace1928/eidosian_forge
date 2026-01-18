import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
@is_player_source.setter
def is_player_source(self, n: bool) -> None:
    self._source.is_player_source = n