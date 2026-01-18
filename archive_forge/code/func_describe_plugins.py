import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def describe_plugins(show_paths=False, state=None):
    """Generate text description of plugins.

    Includes both those that have loaded, and those that failed to load.

    Args:
      show_paths: If true, include the plugin path.
      state: The library state object to inspect.

    Returns:
      Iterator of text lines (including newlines.)
    """
    if state is None:
        state = breezy.get_global_state()
    loaded_plugins = getattr(state, 'plugins', {})
    plugin_warnings = set(getattr(state, 'plugin_warnings', []))
    all_names = sorted(set(loaded_plugins.keys()).union(plugin_warnings))
    for name in all_names:
        if name in loaded_plugins:
            plugin = loaded_plugins[name]
            version = plugin.__version__
            if version == 'unknown':
                version = ''
            yield '{} {}\n'.format(name, version)
            d = plugin.module.__doc__
            if d:
                doc = d.split('\n')[0]
            else:
                doc = '(no description)'
            yield ('  %s\n' % doc)
            if show_paths:
                yield ('   %s\n' % plugin.path())
        else:
            yield ('%s (failed to load)\n' % name)
        if name in state.plugin_warnings:
            for line in state.plugin_warnings[name]:
                yield ('  ** ' + line + '\n')
        yield '\n'