import pytest
from panel.widgets.indicators import (
def test_dial_none(document, comm):
    dial = Dial(value=None, name='Value')
    model = dial.get_root(document, comm)
    cds = model.select(name='annulus_source')
    assert list(cds.data['starts']) == [9.861110273767961, 9.861110273767961]
    assert list(cds.data['ends']) == [9.861110273767961, 5.846852994181004]
    text_cds = model.select(name='label_source')
    assert text_cds.data['text'] == ['Value', '-%', '0%', '100%']
    dial.nan_format = 'nan'
    assert text_cds.data['text'] == ['Value', 'nan%', '0%', '100%']