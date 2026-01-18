import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
def scan_dir_for_imgs(self, directory, glob_pattern='*.tif', img_class=Img):
    """
        Search the given directory for the associated world files
        of the image files.

        Parameters
        ----------
        directory
            The directory path to search for image files.
        glob_pattern: optional
            The image filename glob pattern to search with.
            Defaults to ``'*.tif'``.
        img_class: optional
            The class used to construct each image in the Collection.

        Note
        ----
            Does not recursively search sub-directories.

        """
    imgs = Path(directory).glob(glob_pattern)
    for img in imgs:
        dirname, fname = (img.parent, img.name)
        worlds = img_class.world_files(fname)
        for fworld in worlds:
            fworld = dirname / fworld
            if fworld.exists():
                break
        else:
            raise ValueError(f'Image file {img!r} has no associated world file')
        self.images.append(img_class.from_world_file(img, fworld))