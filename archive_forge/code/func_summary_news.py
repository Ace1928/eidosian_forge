from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
def summary_news(self, sparsify=True):
    """
        Create summary table showing news from new data since previous results

        Parameters
        ----------
        sparsify : bool, optional, default True
            Set to False for the table to include every one of the multiindex
            keys at each row.

        Returns
        -------
        updates_table : SimpleTable
            Table showing new datapoints that were not in the previous results'
            data. Columns are:

            - `update date` : date associated with a new data point.
            - `updated variable` : variable for which new data was added at
              `update date`.
            - `forecast (prev)` : the forecast value for the updated variable
              at the update date in the previous results object (i.e. prior to
              the data being available).
            - `observed` : the observed value of the new datapoint.

        See Also
        --------
        data_updates
        """
    data = pd.merge(self.data_updates, self.news, left_index=True, right_index=True).sort_index().reset_index()
    try:
        data[['update date', 'updated variable']] = data[['update date', 'updated variable']].applymap(str)
        data.iloc[:, 2:] = data.iloc[:, 2:].applymap(lambda num: '' if pd.isnull(num) else '%.2f' % num)
    except AttributeError:
        cols = ['update date', 'updated variable']
        data[cols] = data[cols].map(str)
        cols = data.columns[2:]
        data[cols] = data[cols].map(lambda num: '' if pd.isnull(num) else '%.2f' % num)
    if sparsify:
        mask = data['update date'] == data['update date'].shift(1)
        data.loc[mask, 'update date'] = ''
    params_data = data.values
    params_header = data.columns.tolist()
    params_stubs = None
    title = 'News from updated observations:'
    updates_table = SimpleTable(params_data, params_header, params_stubs, txt_fmt=fmt_params, title=title)
    return updates_table