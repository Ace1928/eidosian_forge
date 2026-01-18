from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
def geometries(self):
    min_x, min_y, max_x, max_y = self.crs.boundary.bounds
    geoms = self.source.fetch_geometries(self.crs, extent=(min_x, max_x, min_y, max_y))
    return iter(geoms)