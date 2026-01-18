from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def differential_expression_by_cluster(data, clusters, measure='difference', direction='both', gene_names=None, n_jobs=-2):
    """Calculate the most significant genes for each cluster in a dataset.

    Measurements are run for each cluster against the rest of the dataset.

    Parameters
    ----------
    data : array-like, shape=[n_cells, n_genes]
    clusters : list-like, shape=[n_cells]
    measure : {'difference', 'emd', 'ttest', 'ranksum'}, optional
              (default: 'difference')
        The measurement to be used to rank genes.
        'difference' is the mean difference between genes.
        'emd' refers to Earth Mover's Distance.
        'ttest' refers to Welch's t-statistic.
        'ranksum' refers to the Wilcoxon rank sum statistic (or the Mann-Whitney U
        statistic).
    direction : {'up', 'down', 'both'}, optional (default: 'both')
        The direction in which to consider genes significant. If 'up', rank genes
        where X > Y. If 'down', rank genes where X < Y. If 'both', rank genes
        by absolute value.
    gene_names : list-like or `None`, optional (default: `None`)
        List of gene names associated with the columns of X and Y
    n_jobs : int, optional (default: -2)
        Number of threads to use if the measurement is parallelizable (currently used
        for EMD). If negative, -1 refers to all available cores.

    Returns
    -------
    result : dict(pd.DataFrame)
        Dictionary containing an ordered DataFrame with a column "gene" and a column
        named `measure` for each cluster.
    """
    if gene_names is not None and isinstance(data, pd.DataFrame):
        data = select.select_cols(data, idx=gene_names)
        gene_names = data.columns
    if gene_names is None:
        if isinstance(data, pd.DataFrame):
            gene_names = data.columns
    elif not len(gene_names) == data.shape[1]:
        raise ValueError('Expected gene_names to have length {}. Got {}'.format(data.shape[1], len(gene_names)))
    data = utils.to_array_or_spmatrix(data)
    result = {cluster: differential_expression(select.select_rows(data, idx=clusters == cluster), select.select_rows(data, idx=clusters != cluster), measure=measure, direction=direction, gene_names=gene_names, n_jobs=n_jobs) for cluster in np.unique(clusters)}
    return result