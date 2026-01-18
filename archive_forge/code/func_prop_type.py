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
def prop_type(text):
    """Returns a schema property's proper Python type.

    Parameters
    ----------
    text : str
        A type name, with or without width.

    Returns
    -------
    obj
        A Python class.

    Examples
    --------
    >>> prop_type('int')
    <class 'int'>
    >>> prop_type('str:25')
    <class 'str'>

    """
    key = text.split(':')[0]
    return FIELD_TYPES_MAP[key]