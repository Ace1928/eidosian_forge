import logging
import os
from types import SimpleNamespace
from rasterio._path import _parse_path, _UnparsedPath
@staticmethod
def aws_or_dummy(*args, **kwargs):
    """Create an AWSSession if boto3 is available, else DummySession

        Parameters
        ----------
        path : str
            A dataset path or identifier.
        args : sequence
            Positional arguments for the foreign session constructor.
        kwargs : dict
            Keyword arguments for the foreign session constructor.

        Returns
        -------
        Session

        """
    if boto3 is not None:
        return AWSSession(*args, **kwargs)
    else:
        return DummySession(*args, **kwargs)