import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def plugin_info(plugin):
    """Return plugin meta-data.

    Parameters
    ----------
    plugin : str
        Name of plugin.

    Returns
    -------
    m : dict
        Meta data as specified in plugin ``.ini``.

    """
    try:
        return plugin_meta_data[plugin]
    except KeyError:
        raise ValueError(f'No information on plugin "{plugin}"')