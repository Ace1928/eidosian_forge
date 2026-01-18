from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_multiline_entries():
    d = TextDiagramDrawer()
    d.write(0, 0, 'hello\nthere')
    d.write(0, 1, 'next')
    d.write(5, 1, '1\n2\n3')
    d.write(5, 2, '4n')
    d.vertical_line(x=5, y1=1, y2=2)
    d.horizontal_line(y=1, x1=0, x2=8)
    _assert_same_diagram(d.render().strip(), '\nhello\nthere\n\n              1\nnext──────────2──────\n              3\n              │\n              4n\n    '.strip())
    d = TextDiagramDrawer()
    d.vertical_line(x=0, y1=0, y2=3)
    d.vertical_line(x=1, y1=0, y2=3)
    d.vertical_line(x=2, y1=0, y2=3)
    d.vertical_line(x=3, y1=0, y2=3)
    d.write(0, 0, 'long line\nshort')
    d.write(2, 2, 'short\nlong line')
    _assert_same_diagram(d.render().strip(), '\nlong line ╷ ╷         ╷\nshort     │ │         │\n│         │ │         │\n│         │ │         │\n│         │ │         │\n│         │ short     │\n│         │ long line │\n│         │ │         │\n    '.strip())