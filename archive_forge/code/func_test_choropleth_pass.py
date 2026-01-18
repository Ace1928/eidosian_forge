import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def test_choropleth_pass(self):
    """Make sure default choropleth pass"""
    self.world.explore(column='pop_est')