from contextlib import contextmanager
import numpy as np
import pytest
from skimage._shared._dependency_checks import has_mpl
from skimage import io
from skimage.io import manage_plugins
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_load_preferred_plugins_all():
    from skimage.io._plugins import pil_plugin, matplotlib_plugin
    with protect_preferred_plugins():
        manage_plugins.preferred_plugins = {'all': ['pil'], 'imshow': ['matplotlib']}
        manage_plugins.reset_plugins()
        for plugin_type in ('imread', 'imsave'):
            plug, func = manage_plugins.plugin_store[plugin_type][0]
            assert func == getattr(pil_plugin, plugin_type)
        plug, func = manage_plugins.plugin_store['imshow'][0]
        assert func == getattr(matplotlib_plugin, 'imshow')