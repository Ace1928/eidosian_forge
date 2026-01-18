import numpy as np
from .core.imopen import imopen
def improps(uri, *, index=None, plugin=None, extension=None, **kwargs):
    """Read standardized metadata.

    Opens the given URI and reads the properties of an ndimage from it. The
    properties represent standardized metadata. This means that they will have
    the same name regardless of the format being read or plugin/backend being
    used. Further, any field will be, where possible, populated with a sensible
    default (may be `None`) if the ImageResource does not declare a value in its
    metadata.

    Parameters
    ----------
    index : int
        If the ImageResource contains multiple ndimages, and index is an
        integer, select the index-th ndimage from among them and return its
        properties. If index is an ellipsis (...), read all ndimages in the file
        and stack them along a new batch dimension and return their properties.
        If index is None, let the plugin decide.
    plugin : {str, None}
        The plugin to be used. If None, performs a search for a matching
        plugin.
    extension : str
        If not None, treat the provided ImageResource as if it had the given
        extension. This affects the order in which backends are considered.
    **kwargs :
        Additional keyword arguments will be passed to the plugin's ``properties``
        call.

    Returns
    -------
    properties : ImageProperties
        A dataclass filled with standardized image metadata.

    Notes
    -----
    Where possible, this will avoid loading pixel data.

    See Also
    --------
    imageio.core.v3_plugin_api.ImageProperties

    """
    plugin_kwargs = {'legacy_mode': False, 'plugin': plugin, 'extension': extension}
    call_kwargs = kwargs
    if index is not None:
        call_kwargs['index'] = index
    with imopen(uri, 'r', **plugin_kwargs) as img_file:
        properties = img_file.properties(**call_kwargs)
    return properties