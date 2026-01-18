import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def test_lines_stretch_content():
    d = BlockDiagramDrawer()
    d.mutable_block(0, 0).left = 'l'
    d.mutable_block(2, 4).right = 'r'
    d.mutable_block(11, 15).bottom = 'b'
    d.mutable_block(16, 17).top = 't'
    d.mutable_block(19, 20).center = 'c'
    d.mutable_block(21, 23).content = 'C'
    _assert_same_diagram(d.render(), '\n\n\n  C'[1:])