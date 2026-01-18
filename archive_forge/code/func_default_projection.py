import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def default_projection(self):
    """
        Return a :class:`cartopy.crs.Projection` in which the WFS
        service can supply the requested features.

        """
    if self._default_urn is None:
        default_urn = {self.service.contents[feature].crsOptions[0] for feature in self.features}
        if len(default_urn) != 1:
            ValueError('Failed to find a single common default SRS across all features (typenames).')
        else:
            default_urn = default_urn.pop()
        if str(default_urn) not in _URN_TO_CRS and ':EPSG:' not in str(default_urn):
            raise ValueError(f'Unknown mapping from SRS/CRS_URN {default_urn!r} to cartopy projection.')
        self._default_urn = default_urn
    if str(self._default_urn) in _URN_TO_CRS:
        return _URN_TO_CRS[str(self._default_urn)]
    elif ':EPSG:' in str(self._default_urn):
        epsg_num = str(self._default_urn).split(':')[-1]
        return ccrs.epsg(int(epsg_num))
    else:
        raise ValueError(f'Unknown coordinate reference system: {str(self._default_urn)}')