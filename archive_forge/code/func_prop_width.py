import glob
import logging
import os
from pathlib import Path
import platform
import warnings
from fiona._env import (
from fiona._env import driver_count
from fiona._show_versions import show_versions
from fiona.collection import BytesCollection, Collection
from fiona.drvsupport import supported_drivers
from fiona.env import ensure_env_with_credentials, Env
from fiona.errors import FionaDeprecationWarning
from fiona.io import MemoryFile
from fiona.model import Feature, Geometry, Properties
from fiona.ogrext import (
from fiona.path import ParsedPath, parse_path, vsi_path
from fiona.vfs import parse_paths as vfs_parse_paths
from fiona import _geometry, _err, rfc3339
import uuid
def prop_width(val):
    """Returns the width of a str type property.

    Undefined for non-str properties.

    Parameters
    ----------
    val : str
        A type:width string from a collection schema.

    Returns
    -------
    int or None

    Examples
    --------
    >>> prop_width('str:25')
    25
    >>> prop_width('str')
    80

    """
    if val.startswith('str'):
        return int((val.split(':')[1:] or ['80'])[0])
    return None