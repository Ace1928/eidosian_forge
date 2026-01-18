from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
def draw_curve(self, grid_characters: BoxDrawCharacterSet, *, top: bool=False, left: bool=False, right: bool=False, bottom: bool=False, crossing_char: Optional[str]=None):
    """Draws lines in the box using the given character set.

        Supports merging the new lines with the lines from a previous call to
        draw_curve, including when they have different character sets (assuming
        there exist characters merging the two).

        Args:
            grid_characters: The character set to draw the curve with.
            top: Draw topward leg?
            left: Draw leftward leg?
            right: Draw rightward leg?
            bottom: Draw downward leg?
            crossing_char: Overrides the all-legs-present character. Useful for
                ascii diagrams, where the + doesn't always look the clearest.
        """
    if not any([top, left, right, bottom]):
        return
    sign_top = +1 if top else -1 if self.top else 0
    sign_bottom = +1 if bottom else -1 if self.bottom else 0
    sign_left = +1 if left else -1 if self.left else 0
    sign_right = +1 if right else -1 if self.right else 0
    if top:
        self.top = grid_characters.top_bottom
    if bottom:
        self.bottom = grid_characters.top_bottom
    if left:
        self.left = grid_characters.left_right
    if right:
        self.right = grid_characters.left_right
    if not all([crossing_char, self.top, self.bottom, self.left, self.right]):
        crossing_char = box_draw_character(self._prev_curve_grid_chars, grid_characters, top=sign_top, bottom=sign_bottom, left=sign_left, right=sign_right)
    self.center = crossing_char or ''
    self._prev_curve_grid_chars = grid_characters