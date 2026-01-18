import warnings
import click
from rasterio.enums import Resampling
from rasterio.errors import RasterioDeprecationWarning
from rasterio.rio import options
from rasterio.rio.helpers import resolve_inout
def deprecated_precision(*args):
    warnings.warn('The --precision option is unused, deprecated, and will be removed in 2.0.0.', RasterioDeprecationWarning)
    return None