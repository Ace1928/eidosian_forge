import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def test_mixed_block_curve():
    diagram = BlockDiagramDrawer()
    for a, b, c, d in itertools.product(range(3), repeat=4):
        x = (a * 3 + b) * 2
        y = (c * 3 + d) * 2
        block = diagram.mutable_block(x, y)
        block.horizontal_alignment = 0.5
        block.draw_curve(NORMAL_BOX_CHARS, top=a == 2, bottom=b == 2, left=c == 2, right=d == 2)
        block.draw_curve(BOLD_BOX_CHARS, top=a == 1, bottom=b == 1, left=c == 1, right=d == 1)
    actual = diagram.render(min_block_width=3, min_block_height=3)
    expected = '\n                   ┃     ┃     ┃     │     │     │\n       ╻     ╷     ╹     ┃     ╿     ╵     ╽     │\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n ╺━    ┏━    ┍━    ┗━    ┣━    ┡━    ┕━    ┢━    ┝━\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n ╶─    ┎─    ┌─    ┖─    ┠─    ┞─    └─    ┟─    ├─\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n━╸    ━┓    ━┑    ━┛    ━┫    ━┩    ━┙    ━┪    ━┥\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n━━━   ━┳━   ━┯━   ━┻━   ━╋━   ━╇━   ━┷━   ━╈━   ━┿━\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n━╾─   ━┱─   ━┭─   ━┹─   ━╉─   ━╃─   ━┵─   ━╅─   ━┽─\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n─╴    ─┒    ─┐    ─┚    ─┨    ─┦    ─┘    ─┧    ─┤\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n─╼━   ─┲━   ─┮━   ─┺━   ─╊━   ─╄━   ─┶━   ─╆━   ─┾━\n       ┃     │           ┃     │           ┃     │\n\n\n\n                   ┃     ┃     ┃     │     │     │\n───   ─┰─   ─┬─   ─┸─   ─╂─   ─╀─   ─┴─   ─╁─   ─┼─\n       ┃     │           ┃     │           ┃     │'[1:]
    _assert_same_diagram(actual, expected)