import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
@classmethod
def hascreds(cls, config):
    """Determine if the given configuration has proper credentials

        Parameters
        ----------
        cls : class
            A Session class.
        config : dict
            GDAL configuration as a dict.

        Returns
        -------
        bool

        """
    return 'AZURE_STORAGE_CONNECTION_STRING' in config or ('AZURE_STORAGE_ACCOUNT' in config and 'AZURE_STORAGE_ACCESS_KEY' in config) or ('AZURE_STORAGE_ACCOUNT' in config and 'AZURE_NO_SIGN_REQUEST' in config)