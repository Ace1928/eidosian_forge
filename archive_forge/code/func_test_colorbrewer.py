import palettable
from palettable.palette import Palette
def test_colorbrewer():
    assert isinstance(palettable.colorbrewer.diverging.PuOr_6, Palette)
    assert isinstance(palettable.colorbrewer.qualitative.Pastel1_9, Palette)
    assert isinstance(palettable.colorbrewer.sequential.PuBuGn_9, Palette)