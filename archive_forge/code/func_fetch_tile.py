from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
def fetch_tile(tile):
    try:
        img, extent, origin = self.get_image(tile)
    except OSError:
        raise
    img = np.array(img)
    x = np.linspace(extent[0], extent[1], img.shape[1])
    y = np.linspace(extent[2], extent[3], img.shape[0])
    return (img, x, y, origin)