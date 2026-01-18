import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def call_plugin(kind, *args, **kwargs):
    """Find the appropriate plugin of 'kind' and execute it.

    Parameters
    ----------
    kind : {'imshow', 'imsave', 'imread', 'imread_collection'}
        Function to look up.
    plugin : str, optional
        Plugin to load.  Defaults to None, in which case the first
        matching plugin is used.
    *args, **kwargs : arguments and keyword arguments
        Passed to the plugin function.

    """
    if kind not in plugin_store:
        raise ValueError(f'Invalid function ({kind}) requested.')
    plugin_funcs = plugin_store[kind]
    if len(plugin_funcs) == 0:
        msg = f'No suitable plugin registered for {kind}.\n\nYou may load I/O plugins with the `skimage.io.use_plugin` command.  A list of all available plugins are shown in the `skimage.io` docstring.'
        raise RuntimeError(msg)
    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        _, func = plugin_funcs[0]
    else:
        _load(plugin)
        try:
            func = [f for p, f in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    return func(*args, **kwargs)