import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
@classmethod
def from_world_file(cls, img_fname, world_fname):
    """
        Return an Img instance from the given image filename and
        worldfile filename.

        """
    im = Image.open(img_fname)
    with open(world_fname) as world_fh:
        extent, pix_size = cls.world_file_extent(world_fh, im.size)
    if hasattr(im, 'close'):
        im.close()
    return cls(img_fname, extent, 'lower', pix_size)