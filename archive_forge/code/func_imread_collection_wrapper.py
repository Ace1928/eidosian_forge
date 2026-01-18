import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
def imread_collection_wrapper(imread):

    def imread_collection(load_pattern, conserve_memory=True):
        """Return an `ImageCollection` from files matching the given pattern.

        Note that files are always stored in alphabetical order. Also note that
        slicing returns a new ImageCollection, *not* a view into the data.

        See `skimage.io.ImageCollection` for details.

        Parameters
        ----------
        load_pattern : str or list
            Pattern glob or filenames to load. The path can be absolute or
            relative.  Multiple patterns should be separated by a colon,
            e.g. ``/tmp/work/*.png:/tmp/other/*.jpg``.  Also see
            implementation notes below.
        conserve_memory : bool, optional
            If True, never keep more than one in memory at a specific
            time.  Otherwise, images will be cached once they are loaded.

        """
        return ImageCollection(load_pattern, conserve_memory=conserve_memory, load_func=imread)
    return imread_collection