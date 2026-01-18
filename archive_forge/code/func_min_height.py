from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
def min_height(self) -> int:
    """Minimum height necessary to render the block's contents."""
    return max(len(self.content.split('\n')) if self.content else 0, int(any([self.left, self.right])))