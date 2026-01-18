import numpy as np
from scipy import stats  #get rid of this? need only norm.sf
def aggregate_raters(data, n_cat=None):
    """convert raw data with shape (subject, rater) to (subject, cat_counts)

    brings data into correct format for fleiss_kappa

    bincount will raise exception if data cannot be converted to integer.

    Parameters
    ----------
    data : array_like, 2-Dim
        data containing category assignment with subjects in rows and raters
        in columns.
    n_cat : None or int
        If None, then the data is converted to integer categories,
        0,1,2,...,n_cat-1. Because of the relabeling only category levels
        with non-zero counts are included.
        If this is an integer, then the category levels in the data are already
        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the
        returned array may contain columns with zero count, if no subject
        has been categorized with this level.

    Returns
    -------
    arr : nd_array, (n_rows, n_cat)
        Contains counts of raters that assigned a category level to individuals.
        Subjects are in rows, category levels in columns.
    categories : nd_array, (n_category_levels,)
        Contains the category levels.

    """
    data = np.asarray(data)
    n_rows = data.shape[0]
    if n_cat is None:
        cat_uni, cat_int = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
    else:
        cat_uni = np.arange(n_cat)
        data_ = data
    tt = np.zeros((n_rows, n_cat), int)
    for idx, row in enumerate(data_):
        ro = np.bincount(row)
        tt[idx, :len(ro)] = ro
    return (tt, cat_uni)