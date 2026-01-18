import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def read_user_background_images(self, verify=True):
    """
        Read the metadata in the specified CARTOPY_USER_BACKGROUNDS
        environment variable to populate the dictionaries for background_img.

        If CARTOPY_USER_BACKGROUNDS is not set then by default the image in
        lib/cartopy/data/raster/natural_earth/ will be made available.

        The metadata should be a standard JSON file which specifies a two
        level dictionary. The first level is the image type.
        For each image type there must be the fields:
        __comment__, __source__ and __projection__
        and then an element giving the filename for each resolution.

        An example JSON file can be found at:
        lib/cartopy/data/raster/natural_earth/images.json

        """
    bgdir = Path(os.getenv('CARTOPY_USER_BACKGROUNDS', config['repo_data_dir'] / 'raster' / 'natural_earth'))
    json_file = bgdir / 'images.json'
    with open(json_file) as js_obj:
        dict_in = json.load(js_obj)
    for img_type in dict_in:
        _USER_BG_IMGS[img_type] = dict_in[img_type]
    if verify:
        required_info = ['__comment__', '__source__', '__projection__']
        for img_type in _USER_BG_IMGS:
            if img_type == '__comment__':
                pass
            else:
                for required in required_info:
                    if required not in _USER_BG_IMGS[img_type]:
                        raise ValueError(f'User background metadata file {json_file!r}, image type {img_type!r}, does not specify metadata item {required!r}')
                for resln in _USER_BG_IMGS[img_type]:
                    if resln not in required_info:
                        img_it_r = _USER_BG_IMGS[img_type][resln]
                        test_file = bgdir / img_it_r
                        if not test_file.is_file():
                            raise ValueError(f'File "{test_file}" not found')