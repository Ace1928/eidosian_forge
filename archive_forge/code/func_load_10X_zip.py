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
def load_10X_zip(filename, sparse=True, gene_labels='symbol', allow_duplicates=None):
    """Load zipped 10X data produced from the 10X Cellranger pipeline.

    Runs `load_10X` after unzipping the data contained in `filename`.

    Parameters
    ----------
    filename: string
        path to zipped input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.tsv' to be present and
        will raise an error otherwise
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
    if not os.path.isfile(filename):
        with tempfile.TemporaryDirectory() as download_dir:
            zip_filename = os.path.join(download_dir, 'download.zip')
            try:
                with urllib.request.urlopen(filename) as url:
                    with open(zip_filename, 'wb') as handle:
                        handle.write(url.read())
            except ValueError as e:
                if str(e).startswith('unknown url type:'):
                    raise FileNotFoundError("No such file: '{}'".format(filename))
                else:
                    raise
            else:
                return load_10X_zip(zip_filename, sparse=sparse, gene_labels=gene_labels, allow_duplicates=allow_duplicates)
    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(filename) as handle:
        files = handle.namelist()
        if len(files) < 3:
            valid_dirnames = []
        else:
            valid_dirnames = []
            for dirname in set([''] + ['/'.join(f.split('/')[:-1]) for f in files]):
                subdir_files = [f for f in files if f.startswith(dirname)]

                def path(fn, dirname):
                    if dirname != '':
                        path = '{}/{}'.format(dirname, fn)
                    else:
                        path = fn
                    return path
                if (path('barcodes.tsv', dirname) in subdir_files or path('barcodes.tsv.gz', dirname) in subdir_files) and ((path('genes.tsv', dirname) in subdir_files or path('genes.tsv.gz', dirname) in subdir_files) or (path('features.tsv', dirname) in subdir_files or path('features.tsv.gz', dirname) in subdir_files)) and (path('matrix.mtx', dirname) in subdir_files or path('matrix.mtx.gz', dirname) in subdir_files):
                    valid_dirnames.append(dirname)
        if len(valid_dirnames) != 1:
            raise ValueError("Expected a single zipped folder containing 'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)'. Got {}".format(files))
        dirname = valid_dirnames[0]
        handle.extractall(path=tmpdir)
    data = load_10X(os.path.join(tmpdir, dirname))
    shutil.rmtree(tmpdir)
    return data