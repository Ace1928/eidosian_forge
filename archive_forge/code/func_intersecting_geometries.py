from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
def intersecting_geometries(self, extent):
    geoms = self.source.fetch_geometries(self.crs, extent)
    return iter(geoms)