import pytest
from panel.widgets.indicators import (
def test_dial_thresholds_with_bounds(document, comm):
    dial = Dial(value=25, colors=[(0.33, 'green'), (0.66, 'yellow'), (1, 'red')], bounds=(25, 75))
    model = dial.get_root(document, comm)
    cds = model.select(name='annulus_source')
    assert ['green', 'whitesmoke'] == cds.data['color']
    dial.value = 50
    assert ['yellow', 'whitesmoke'] == cds.data['color']
    dial.value = 75
    assert ['red', 'whitesmoke'] == cds.data['color']