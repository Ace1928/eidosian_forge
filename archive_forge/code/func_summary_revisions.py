from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def summary_revisions(self, sparsify=True):
    """
        Create summary table showing revisions to the previous results' data

        Parameters
        ----------
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.

        Returns
        -------
        revisions_table : SimpleTable
            Table showing revisions to the previous results' data. Columns are:

            - `revision date` : date associated with a revised data point
            - `revised variable` : variable that was revised at `revision date`
            - `observed (prev)` : the observed value prior to the revision
            - `revised` : the new value after the revision
            - `revision` : the new value after the revision
            - `detailed impacts computed` : whether detailed impacts were
              computed for this revision
        """
    data = pd.merge(self.data_revisions, self.revisions_all, left_index=True, right_index=True).sort_index().reset_index()
    data = data[['revision date', 'revised variable', 'observed (prev)', 'revision', 'detailed impacts computed']]
    try:
        data[['revision date', 'revised variable']] = data[['revision date', 'revised variable']].applymap(str)
        data.iloc[:, 2:-1] = data.iloc[:, 2:-1].applymap(lambda num: '' if pd.isnull(num) else '%.2f' % num)
    except AttributeError:
        cols = ['revision date', 'revised variable']
        data[cols] = data[cols].map(str)
        cols = data.columns[2:-1]
        data[cols] = data[cols].map(lambda num: '' if pd.isnull(num) else '%.2f' % num)
    if sparsify:
        mask = data['revision date'] == data['revision date'].shift(1)
        data.loc[mask, 'revision date'] = ''
    params_data = data.values
    params_header = data.columns.tolist()
    params_stubs = None
    title = 'Revisions to dataset:'
    revisions_table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
    return revisions_table