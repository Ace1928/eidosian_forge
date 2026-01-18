import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def test_content_stretches_other_blocks():
    d = BlockDiagramDrawer()
    d.mutable_block(0, 0).horizontal_alignment = 0.5
    d.mutable_block(1, 0).horizontal_alignment = 0.5
    d.mutable_block(0, 1).horizontal_alignment = 0.5
    d.mutable_block(1, 1).horizontal_alignment = 0.5
    d.mutable_block(0, 0).content = 'long text\nwith multiple lines'
    d.mutable_block(1, 0).draw_curve(NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    d.mutable_block(1, 1).draw_curve(NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    d.mutable_block(0, 1).draw_curve(NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    _assert_same_diagram(d.render(), '\n     long text     ┼\nwith multiple lines│\n─────────┼─────────┼'[1:])