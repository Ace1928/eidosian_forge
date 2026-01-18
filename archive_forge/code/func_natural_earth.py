import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
def natural_earth(resolution='110m', category='physical', name='coastline'):
    """
    Return the path to the requested natural earth shapefile,
    downloading and unzipping if necessary.

    To identify valid components for this function, either browse
    NaturalEarthData.com, or if you know what you are looking for, go to
    https://github.com/nvkelso/natural-earth-vector/ to see the actual
    files which will be downloaded.

    Note
    ----
        Some of the Natural Earth shapefiles have special features which are
        described in the name. For example, the 110m resolution
        "admin_0_countries" data also has a sibling shapefile called
        "admin_0_countries_lakes" which excludes lakes in the country
        outlines. For details of what is available refer to the Natural Earth
        website, and look at the "download" link target to identify
        appropriate names.

    """
    ne_downloader = Downloader.from_config(('shapefiles', 'natural_earth', resolution, category, name))
    format_dict = {'config': config, 'category': category, 'name': name, 'resolution': resolution}
    return ne_downloader.path(format_dict)