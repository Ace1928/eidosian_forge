from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def pick_charset(use_unicode: bool, emphasize: bool, doubled: bool) -> BoxDrawCharacterSet:
    if not use_unicode:
        return ASCII_BOX_CHARS
    if emphasize and doubled:
        raise ValueError('Cannot use both emphasized and doubled.')
    if emphasize:
        return BOLD_BOX_CHARS
    if doubled:
        return DOUBLED_BOX_CHARS
    return NORMAL_BOX_CHARS