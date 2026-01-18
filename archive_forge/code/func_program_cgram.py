from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def program_cgram(self, index: int, data: Sequence[int]) -> None:
    """
        Program character data.

        Characters available as chr(0) through chr(7), and repeated as chr(8) through chr(15).

        index -- 0 to 7 index of character to program

        data -- list of 8, 6-bit integer values top to bottom with MSB on the left side of the character.
        """
    if not 0 <= index <= 7:
        raise ValueError(index)
    if len(data) != 8:
        raise ValueError(data)
    self.queue_command(self.CMD_CGRAM, bytearray([index]) + bytearray(data))