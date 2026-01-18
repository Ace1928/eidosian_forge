from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
import logging
def multigroup(pvals, groups, exact=True, keep_all=True, alpha=0.05):
    """Test if the given groups are different from the total partition.

    Given a boolean array test if each group has a proportion of positives
    different than the complexive proportion.
    The test can be done as an exact Fisher test or approximated as a
    Chi squared test for more speed.

    Parameters
    ----------
    pvals : pandas series of boolean
        the significativity of the variables under analysis
    groups : dict of list
        the name of each category of variables under exam.
        each one is a list of the variables included
    exact : bool, optional
        If True (default) use the fisher exact test, otherwise
        use the chi squared test for contingencies tables.
        For high number of elements in the array the fisher test can
        be significantly slower than the chi squared.
    keep_all : bool, optional
        if False it will drop those groups where the fraction
        of positive is below the expected result. If True (default)
         it will keep all the significant results.
    alpha : float, optional
        the significativity level for the pvalue correction
        on the whole set of groups (not inside the groups themselves).

    Returns
    -------
    result_df: pandas dataframe
        for each group returns:

            pvals - the fisher p value of the test
            adj_pvals - the adjusted pvals
            increase - the log of the odd ratio between the
                internal significant ratio versus the external one
            _in_sign - significative elements inside the group
            _in_non - non significative elements inside the group
            _out_sign - significative elements outside the group
            _out_non - non significative elements outside the group

    Notes
    -----
    This test allow to see if a category of variables is generally better
    suited to be described for the model. For example to see if a predictor
    gives more information on demographic or economical parameters,
    by creating two groups containing the endogenous variables of each
    category.

    This function is conceived for medical dataset with a lot of variables
    that can be easily grouped into functional groups. This is because
    The significativity of a group require a rather large number of
    composing elements.

    Examples
    --------
    A toy example on a real dataset, the Guerry dataset from R
    >>> url = "https://raw.githubusercontent.com/vincentarelbundock/"
    >>> url = url + "Rdatasets/csv/HistData/Guerry.csv"
    >>> df = pd.read_csv(url, index_col='dept')

    evaluate the relationship between the various paramenters whith the Wealth
    >>> pvals = multiOLS('Wealth', df)['adj_pvals', '_f_test']

    define the groups
    >>> groups = {}
    >>> groups['crime'] = ['Crime_prop', 'Infanticide',
    ...     'Crime_parents', 'Desertion', 'Crime_pers']
    >>> groups['religion'] = ['Donation_clergy', 'Clergy', 'Donations']
    >>> groups['wealth'] = ['Commerce', 'Lottery', 'Instruction', 'Literacy']

    do the analysis of the significativity
    >>> multigroup(pvals < 0.05, groups)
    """
    pvals = pd.Series(pvals)
    if not set(pvals.unique()) <= {False, True}:
        raise ValueError('the series should be binary')
    if hasattr(pvals.index, 'is_unique') and (not pvals.index.is_unique):
        raise ValueError('series with duplicated index is not accepted')
    results = {'pvals': {}, 'increase': {}, '_in_sign': {}, '_in_non': {}, '_out_sign': {}, '_out_non': {}}
    for group_name, group_list in groups.items():
        res = _test_group(pvals, group_name, group_list, exact)
        results['pvals'][group_name] = res[0]
        results['increase'][group_name] = res[1]
        results['_in_sign'][group_name] = res[2][0]
        results['_in_non'][group_name] = res[2][1]
        results['_out_sign'][group_name] = res[2][2]
        results['_out_non'][group_name] = res[2][3]
    result_df = pd.DataFrame(results).sort_values('pvals')
    if not keep_all:
        result_df = result_df[result_df.increase]
    smt = stats.multipletests
    corrected = smt(result_df['pvals'], method='fdr_bh', alpha=alpha)[1]
    result_df['adj_pvals'] = corrected
    return result_df