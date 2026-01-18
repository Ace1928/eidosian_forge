from . import hdf5
from .utils import _matrix_to_data_frame
import numpy as np
import os
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import shutil
import tempfile
import urllib
import warnings
import zipfile
def load_10X(data_dir, sparse=True, gene_labels='symbol', allow_duplicates=None):
    """Load data produced from the 10X Cellranger pipeline.

    A default run of the `cellranger count` command will generate gene-barcode
    matrices for secondary analysis. For both "raw" and "filtered" output,
    directories are created containing three files:
    'matrix.mtx', 'barcodes.tsv', 'genes.tsv'.
    Running `scprep.io.load_10X(data_dir)` will return a Pandas DataFrame with
    genes as columns and cells as rows.

    Parameters
    ----------
    data_dir: string
        path to input data directory
        expects 'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', 'barcodes.tsv(.gz)'
        to be present and will raise an error otherwise
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.
    allow_duplicates : bool, optional (default: None)
        Whether or not to allow duplicate gene names. If None, duplicates are
        allowed for dense input but not for sparse input.

    Returns
    -------
    data: array-like, shape=[n_samples, n_features]
        If sparse, data will be a pd.DataFrame[pd.SparseArray]. Otherwise, data will
        be a pd.DataFrame.
    """
    if gene_labels not in ['id', 'symbol', 'both']:
        raise ValueError("gene_labels='{}' not recognized. Choose from ['symbol', 'id', 'both']".format(gene_labels))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError('{} is not a directory'.format(data_dir))
    try:
        m = sio.mmread(_find_gz_file(data_dir, 'matrix.mtx'))
        try:
            genes = pd.read_csv(_find_gz_file(data_dir, 'genes.tsv'), delimiter='\t', header=None)
        except FileNotFoundError:
            genes = pd.read_csv(_find_gz_file(data_dir, 'features.tsv'), delimiter='\t', header=None)
        if genes.shape[1] == 2:
            genes.columns = ['id', 'symbol']
        else:
            genes.columns = ['id', 'symbol', 'measurement']
        barcodes = pd.read_csv(_find_gz_file(data_dir, 'barcodes.tsv'), delimiter='\t', header=None)
    except (FileNotFoundError, IOError):
        raise FileNotFoundError("'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)' must be present in {}".format(data_dir))
    cell_names = barcodes[0]
    if allow_duplicates is None:
        allow_duplicates = not sparse
    gene_names = _parse_10x_genes(genes['symbol'].values.astype(str), genes['id'].values.astype(str), gene_labels=gene_labels, allow_duplicates=allow_duplicates)
    data = _matrix_to_data_frame(m.T, cell_names=cell_names, gene_names=gene_names, sparse=sparse)
    return data