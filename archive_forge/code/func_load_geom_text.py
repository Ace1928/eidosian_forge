import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def load_geom_text(x):
    """Load from binary encoded as text."""
    return shapely.wkb.loads(str(x), hex=True)