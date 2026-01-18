import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def list_profiles_in(path):
    """list profiles in a given root directory"""
    profiles = []
    files = os.scandir(path)
    for f in files:
        if f.is_dir() and f.name.startswith('profile_'):
            profiles.append(f.name.split('_', 1)[-1])
    return profiles