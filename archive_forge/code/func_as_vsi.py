import pathlib
import re
import sys
from urllib.parse import urlparse
import attr
from rasterio.errors import PathError
def as_vsi(self):
    return _vsi_path(self)