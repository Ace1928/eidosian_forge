import palettable
from palettable.palette import Palette
def test_cubehelix():
    assert isinstance(palettable.cubehelix.classic_16, Palette)