from __future__ import annotations
from . import Image, ImageFile
from ._binary import o8
from ._binary import o16be as o16b
def build_prototype_image():
    image = Image.new('L', (1, len(_Palm8BitColormapValues)))
    image.putdata(list(range(len(_Palm8BitColormapValues))))
    palettedata = ()
    for colormapValue in _Palm8BitColormapValues:
        palettedata += colormapValue
    palettedata += (0, 0, 0) * (256 - len(_Palm8BitColormapValues))
    image.putpalette(palettedata)
    return image