import logging
from io import BytesIO
from os import PathLike, makedirs, remove
from os.path import exists
import joblib
import numpy as np
from ..utils import Bunch
from ..utils._param_validation import validate_params
from . import get_data_home
from ._base import RemoteFileMetadata, _fetch_remote, _pkl_filepath
def construct_grids(batch):
    """Construct the map grid from the batch object

    Parameters
    ----------
    batch : Batch object
        The object returned by :func:`fetch_species_distributions`

    Returns
    -------
    (xgrid, ygrid) : 1-D arrays
        The grid corresponding to the values in batch.coverages
    """
    xmin = batch.x_left_lower_corner + batch.grid_size
    xmax = xmin + batch.Nx * batch.grid_size
    ymin = batch.y_left_lower_corner + batch.grid_size
    ymax = ymin + batch.Ny * batch.grid_size
    xgrid = np.arange(xmin, xmax, batch.grid_size)
    ygrid = np.arange(ymin, ymax, batch.grid_size)
    return (xgrid, ygrid)