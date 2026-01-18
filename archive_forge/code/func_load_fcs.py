from .. import utils
from .._lazyload import fcsparser
from .utils import _matrix_to_data_frame
from io import BytesIO
import numpy as np
import pandas as pd
import string
import struct
import warnings
@utils._with_pkg(pkg='fcsparser')
def load_fcs(filename, gene_names=True, cell_names=True, sparse=None, metadata_channels=['Time', 'Event_length', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'bead1'], channel_naming='$PnS', reformat_meta=True, override=False, **kwargs):
    """Load a fcs file.

    Parameters
    ----------
    filename : str
        The name of the fcs file to be loaded
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are contained in the file. Otherwise
        expects a filename or an array containing a list of gene symbols or ids
    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are contained in the file. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: None)
        If True, loads the data as a pd.DataFrame[SparseArray]. This uses less memory
        but more CPU.
    metadata_channels : list-like, optional, shape=[n_meta]
        (default: ['Time', 'Event_length', 'DNA1',
            'DNA2', 'Cisplatin', 'beadDist', 'bead1'])
        Channels to be excluded from the data
    channel_naming: '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)
        $PnN stands for the short name (guaranteed to be unique). Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique). Will look like
        'FSC-H' (Forward scatter)
        The chosen field will be used to population self.channels
        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS
        specification.
    reformat_meta : bool, optional (default: True)
        If true, the meta data is reformatted with the channel information
        organized into a DataFrame and moved into the '_channels_' key
    override : bool, optional (default: False)
        If true, uses an experimental override of fcsparser. Should only be
        used in cases where fcsparser fails to load the file, likely due to
        a malformed header. Credit to https://github.com/pontikos/fcstools
    **kwargs : optional arguments for `fcsparser.parse`.

    Returns
    -------
    channel_metadata : dict
        FCS metadata
    cell_metadata : array-like, shape=[n_samples, n_meta]
        Values from metadata channels
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.DataFrame[SparseArray]. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    if cell_names is True:
        cell_names = None
    if gene_names is True:
        gene_names = None
    if override:
        channel_metadata, data = _fcsextract(filename, reformat_meta=reformat_meta, channel_naming=channel_naming, **kwargs)
    else:
        try:
            channel_metadata, data = fcsparser.api.parse(filename, reformat_meta=reformat_meta, **kwargs)
        except (fcsparser.api.ParserFeatureNotImplementedError, ValueError):
            raise RuntimeError("fcsparser failed to load {}, likely due to a malformed header. You can try using `override=True` to use scprep's built-in experimental FCS parser.".format(filename))
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    cell_metadata = data[metadata_channels]
    data = data[data_channels]
    data = _matrix_to_data_frame(data, gene_names=gene_names, cell_names=cell_names, sparse=sparse)
    return (channel_metadata, cell_metadata, data)