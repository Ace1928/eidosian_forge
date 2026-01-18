from os.path import join, dirname, exists, abspath
from kivy import kivy_data_dir
from kivy.cache import Cache
from kivy.utils import platform
from kivy.logger import Logger
import sys
import os
import kivy
def resource_find(filename, use_cache='KIVY_DOC_INCLUDE' not in os.environ):
    """Search for a resource in the list of paths.
    Use resource_add_path to add a custom path to the search.
    By default, results are cached for 60 seconds.
    This can be disabled using use_cache=False.

    .. versionchanged:: 2.1.0
        `use_cache` parameter added and made True by default.
    """
    if not filename:
        return
    found_filename = None
    if use_cache:
        found_filename = Cache.get('kv.resourcefind', filename)
        if found_filename:
            return found_filename
    if filename[:8] == 'atlas://':
        found_filename = filename
    else:
        abspath_filename = abspath(filename)
        if exists(abspath_filename):
            found_filename = abspath(filename)
        else:
            for path in reversed(resource_paths):
                abspath_filename = abspath(join(path, filename))
                if exists(abspath_filename):
                    found_filename = abspath_filename
                    break
        if not found_filename and filename.startswith('data:'):
            found_filename = filename
    if use_cache and found_filename:
        Cache.append('kv.resourcefind', filename, found_filename)
    return found_filename