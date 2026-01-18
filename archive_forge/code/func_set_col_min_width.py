from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
def set_col_min_width(self, x: int, min_width: int):
    """Sets a minimum width for blocks in the column with coordinate x."""
    if x < 0:
        raise IndexError('x < 0')
    self._min_widths[x] = min_width