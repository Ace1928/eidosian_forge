from __future__ import unicode_literals
import os.path  # splitext
import pkg_resources
from pybtex.exceptions import PybtexError
def enumerate_plugin_names(plugin_group):
    """Enumerate all plugin names for the given *plugin_group*."""
    return (entry_point.name for entry_point in pkg_resources.iter_entry_points(plugin_group))