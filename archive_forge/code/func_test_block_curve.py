import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def test_block_curve():
    d = _curve_pieces_diagram(NORMAL_BOX_CHARS)
    actual = d.render(min_block_width=5, min_block_height=5)
    expected = '\n\n            ╶──     ──╴       ─────\n\n\n\n\n\n\n\n  │         │         │         │\n  │         │         │         │\n  ╵         └──     ──┘       ──┴──\n\n\n\n\n\n\n\n\n\n  ╷         ┌──     ──┐       ──┬──\n  │         │         │         │\n  │         │         │         │\n\n\n\n\n\n  │         │         │         │\n  │         │         │         │\n  │         ├──     ──┤       ──┼──\n  │         │         │         │\n  │         │         │         │'
    _assert_same_diagram(actual, expected)
    d = _curve_pieces_diagram(DOUBLED_BOX_CHARS)
    actual = d.render(min_block_width=3, min_block_height=3)
    expected = '\n       ══   ══    ═══\n\n\n\n\n ║     ║     ║     ║\n ║     ╚═   ═╝    ═╩═\n\n\n\n\n\n ║     ╔═   ═╗    ═╦═\n ║     ║     ║     ║\n\n\n\n ║     ║     ║     ║\n ║     ╠═   ═╣    ═╬═\n ║     ║     ║     ║'
    _assert_same_diagram(actual, expected)
    d = _curve_pieces_diagram(BOLD_BOX_CHARS)
    actual = d.render(min_block_width=4, min_block_height=4)
    expected = '\n         ╺━━    ━╸      ━━━━\n\n\n\n\n\n\n ┃       ┃       ┃       ┃\n ╹       ┗━━    ━┛      ━┻━━\n\n\n\n\n\n\n\n ╻       ┏━━    ━┓      ━┳━━\n ┃       ┃       ┃       ┃\n ┃       ┃       ┃       ┃\n\n\n\n\n ┃       ┃       ┃       ┃\n ┃       ┣━━    ━┫      ━╋━━\n ┃       ┃       ┃       ┃\n ┃       ┃       ┃       ┃'
    _assert_same_diagram(actual, expected)
    d = _curve_pieces_diagram(ASCII_BOX_CHARS)
    actual = d.render(min_block_width=3, min_block_height=3)
    expected = '\n        -   -     ---\n\n\n\n\n |     |     |     |\n       \\-   -/    -+-\n\n\n\n\n\n       /-   -\\    -+-\n |     |     |     |\n\n\n\n |     |     |     |\n |     +-   -+    -+-\n |     |     |     |'
    _assert_same_diagram(actual, expected)