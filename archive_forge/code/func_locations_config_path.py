import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def locations_config_path():
    """Return per-user configuration ini file filename."""
    return osutils.pathjoin(config_dir(), 'locations.conf')