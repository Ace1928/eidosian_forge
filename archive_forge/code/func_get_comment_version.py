from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
@staticmethod
def get_comment_version(comments: Sequence[str]) -> Tuple[int, int]:
    """Get the version of SDT-control metadata encoded in the comments

        Parameters
        ----------
        comments
            List of SPE file comments, typically ``metadata["comments"]``.

        Returns
        -------
        Major and minor version. ``-1, -1`` if detection failed.
        """
    if comments[4][70:76] != 'COMVER':
        return (-1, -1)
    try:
        return (int(comments[4][76:78]), int(comments[4][78:80]))
    except ValueError:
        return (-1, -1)