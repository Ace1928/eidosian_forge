import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
@classmethod
def from_configuration(cls, name, crs, name_dir_pairs, glob_pattern='*.tif', img_class=Img):
    """
        Create a :class:`~cartopy.io.img_nest.NestedImageCollection` instance
        given the list of image collection name and directory path pairs.

        This is very convenient functionality for simple configuration level
        creation of this complex object.

        For example, to produce a nested collection of OS map tiles::

            files = [['OS 1:1,000,000', '/directory/to/1_to_1m'],
                     ['OS 1:250,000', '/directory/to/1_to_250k'],
                     ['OS 1:50,000', '/directory/to/1_to_50k'],
                    ]
            r = NestedImageCollection.from_configuration('os',
                                                         ccrs.OSGB(),
                                                         files)

        Parameters
        ----------
        name
            The name for the
            :class:`~cartopy.io.img_nest.NestedImageCollection` instance.
        crs
            The :class:`~cartopy.crs.Projection` of the image collection.
        name_dir_pairs
            A list of image collection name and directory path pairs.
        glob_pattern: optional
            The image collection filename glob pattern. Defaults
            to ``'*.tif'``.
        img_class: optional
            The class of images created in the image collection.

        Returns
        -------
        A :class:`~cartopy.io.img_nest.NestedImageCollection` instance.

        Warnings
        --------
            The list of image collection name and directory path pairs must be
            given in increasing resolution order i.e. from low resolution to
            high resolution.

        """
    collections = []
    for collection_name, collection_dir in name_dir_pairs:
        collection = ImageCollection(collection_name, crs)
        collection.scan_dir_for_imgs(collection_dir, glob_pattern=glob_pattern, img_class=img_class)
        collections.append(collection)
    return cls(name, crs, collections)