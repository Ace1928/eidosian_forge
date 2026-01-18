from .. import utils
from .utils import _matrix_to_data_frame
from scipy import sparse
import os
import pandas as pd
import scipy.io as sio
def save_mtx(data, destination, cell_names=None, gene_names=None):
    """Save a mtx file.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data, saved to destination/matrix.mtx
    destination : str
        Directory in which to save the data
    cell_names : list-like, shape=[n_samples], optional (default: None)
        Cell names associated with rows, saved to destination/cell_names.tsv.
        If `data` is a pandas DataFrame and `cell_names` is None,
        these are autopopulated from `data.index`.
    gene_names : list-like, shape=[n_features], optional (default: None)
        Cell names associated with rows, saved to destination/gene_names.tsv.
        If `data` is a pandas DataFrame and `gene_names` is None,
        these are autopopulated from `data.columns`.

    Examples
    --------
    >>> import scprep
    >>> scprep.io.save_mtx(data, destination="my_data")
    >>> reload = scprep.io.load_mtx("my_data/matrix.mtx",
    ...                             cell_names="my_data/cell_names.tsv",
    ...                             gene_names="my_data/gene_names.tsv")
    """
    if isinstance(data, pd.DataFrame):
        if cell_names is None:
            cell_names = data.index
        if gene_names is None:
            gene_names = data.columns
    data = utils.to_array_or_spmatrix(data)
    data = sparse.coo_matrix(data)
    destination = os.path.expanduser(destination)
    if not os.path.isdir(destination):
        os.mkdir(destination)
    if cell_names is not None:
        with open(os.path.join(destination, 'cell_names.tsv'), 'w') as handle:
            for name in cell_names:
                handle.write('{}\n'.format(name))
    if gene_names is not None:
        with open(os.path.join(destination, 'gene_names.tsv'), 'w') as handle:
            for name in gene_names:
                handle.write('{}\n'.format(name))
    sio.mmwrite(os.path.join(destination, 'matrix.mtx'), data)