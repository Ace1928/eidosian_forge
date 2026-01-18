import threading
import warnings
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path
from typing import Any, Optional, Union, overload
from pyproj import CRS
from pyproj._compat import cstrencode
from pyproj._crs import AreaOfUse, CoordinateOperation
from pyproj._datadir import _clear_proj_error
from pyproj._transformer import (  # noqa: F401 pylint: disable=unused-import
from pyproj.datadir import get_user_data_dir
from pyproj.enums import ProjVersion, TransformDirection, WktVersion
from pyproj.exceptions import ProjError
from pyproj.sync import _download_resource_file
from pyproj.utils import _convertback, _copytobuffer
def download_grids(self, directory: Optional[Union[str, Path]]=None, open_license: bool=True, verbose: bool=False) -> None:
    """
        .. versionadded:: 3.0.0

        Download missing grids that can be downloaded automatically.

        .. warning:: There are cases where the URL to download the grid is missing.
                     In those cases, you can enable enable
                     :ref:`debugging-internal-proj` and perform a
                     transformation. The logs will show the grids PROJ searches for.

        Parameters
        ----------
        directory: str or Path, optional
            The directory to download the grids to.
            Defaults to :func:`pyproj.datadir.get_user_data_dir`
        open_license: bool, default=True
            If True, will only download grids with an open license.
        verbose: bool, default=False
            If True, will print information about grids downloaded.
        """
    if directory is None:
        directory = get_user_data_dir(True)
    for unavailable_operation in self.unavailable_operations:
        for grid in unavailable_operation.grids:
            if not grid.available and grid.url.endswith(grid.short_name) and grid.direct_download and (grid.open_license or not open_license):
                _download_resource_file(file_url=grid.url, short_name=grid.short_name, directory=directory, verbose=verbose)
            elif not grid.available and verbose:
                warnings.warn(f'Skipped: {grid}')