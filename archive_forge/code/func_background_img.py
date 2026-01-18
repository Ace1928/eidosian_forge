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
def background_img(self, name='ne_shaded', resolution='low', extent=None, cache=False):
    """
        Add a background image to the map, from a selection of pre-prepared
        images held in a directory specified by the CARTOPY_USER_BACKGROUNDS
        environment variable. That directory is checked with
        func:`self.read_user_background_images` and needs to contain a JSON
        file which defines for the image metadata.

        Parameters
        ----------
        name: optional
            The name of the image to read according to the contents
            of the JSON file. A typical file might have, for instance:
            'ne_shaded' : Natural Earth Shaded Relief
            'ne_grey' : Natural Earth Grey Earth.
        resolution: optional
            The resolution of the image to read, according to
            the contents of the JSON file. A typical file might
            have the following for each name of the image:
            'low', 'med', 'high', 'vhigh', 'full'.
        extent: optional
            Using a high resolution background image zoomed into
            a small area will take a very long time to render as
            the image is prepared globally, even though only a small
            area is used. Adding the extent will only render a
            particular geographic region. Specified as
            [longitude start, longitude end,
            latitude start, latitude end].

                  e.g. [-11, 3, 48, 60] for the UK
                  or [167.0, 193.0, 47.0, 68.0] to cross the date line.

        cache: optional
            Logical flag as to whether or not to cache the loaded
            images into memory. The images are stored before the
            extent is used.

        """
    if len(_USER_BG_IMGS) == 0:
        self.read_user_background_images()
    bgdir = Path(os.getenv('CARTOPY_USER_BACKGROUNDS', config['repo_data_dir'] / 'raster' / 'natural_earth'))
    try:
        fname = _USER_BG_IMGS[name][resolution]
    except KeyError:
        raise ValueError(f'Image {name!r} and resolution {resolution!r} are not present in the user background image metadata in directory {bgdir!r}')
    fpath = bgdir / fname
    if cache:
        if fname in _BACKG_IMG_CACHE:
            img = _BACKG_IMG_CACHE[fname]
        else:
            img = imread(fpath)
            _BACKG_IMG_CACHE[fname] = img
    else:
        img = imread(fpath)
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    if _USER_BG_IMGS[name]['__projection__'] == 'PlateCarree':
        source_proj = ccrs.PlateCarree()
    else:
        raise NotImplementedError('Background image projection undefined')
    if extent is None:
        return self.imshow(img, origin='upper', transform=source_proj, extent=[-180, 180, -90, 90])
    else:
        d_lat = 180 / img.shape[0]
        d_lon = 360 / img.shape[1]
        lat_pts = np.arange(img.shape[0]) * -d_lat - d_lat / 2 + 90
        lon_pts = np.arange(img.shape[1]) * d_lon + d_lon / 2 - 180
        lat_in_range = np.logical_and(lat_pts >= extent[2], lat_pts <= extent[3])
        if extent[0] < 180 and extent[1] > 180:
            lon_in_range1 = np.logical_and(lon_pts >= extent[0], lon_pts <= 180.0)
            img_subset1 = img[lat_in_range, :, :][:, lon_in_range1, :]
            lon_in_range2 = lon_pts + 360.0 <= extent[1]
            img_subset2 = img[lat_in_range, :, :][:, lon_in_range2, :]
            img_subset = np.concatenate((img_subset1, img_subset2), axis=1)
            ret_extent = [lon_pts[lon_in_range1][0] - d_lon / 2, lon_pts[lon_in_range2][-1] + d_lon / 2 + 360, lat_pts[lat_in_range][-1] - d_lat / 2, lat_pts[lat_in_range][0] + d_lat / 2]
        else:
            lon_in_range = np.logical_and(lon_pts >= extent[0], lon_pts <= extent[1])
            img_subset = img[lat_in_range, :, :][:, lon_in_range, :]
            ret_extent = [lon_pts[lon_in_range][0] - d_lon / 2.0, lon_pts[lon_in_range][-1] + d_lon / 2.0, lat_pts[lat_in_range][-1] - d_lat / 2.0, lat_pts[lat_in_range][0] + d_lat / 2.0]
        return self.imshow(img_subset, origin='upper', transform=source_proj, extent=ret_extent)