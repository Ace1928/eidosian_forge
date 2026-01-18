from pathlib import Path
import numpy as np
from ..config import known_extensions
from .request import InitializationError, IOMode
from .v3_plugin_api import ImageProperties, PluginV3
def legacy_get_reader(self, **kwargs):
    """legacy_get_reader(**kwargs)

        a utility method to provide support vor the V2.9 API

        Parameters
        ----------
        kwargs : ...
            Further keyword arguments are passed to the reader. See :func:`.help`
            to see what arguments are available for a particular format.
        """
    self._request._kwargs = kwargs
    try:
        assert Path(self._request.filename).is_dir()
    except OSError:
        pass
    except AssertionError:
        pass
    else:
        return self._format.get_reader(self._request)
    self._request.get_file().seek(0)
    return self._format.get_reader(self._request)