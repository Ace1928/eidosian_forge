import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
def get_queue_source(self) -> 'SourceGroup':
    return self