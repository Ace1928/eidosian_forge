from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def data_revisions(self):
    """
        Revisions to data points that existed in the previous dataset

        Returns
        -------
        data_revisions : pd.DataFrame
            Index is as MultiIndex consisting of `revision date` and
            `revised variable`. The columns are:

            - `observed (prev)`: the value of the data as it was observed
              in the previous dataset.
            - `revised`: the revised value of the data, as it is observed
              in the new dataset
            - `detailed impacts computed`: whether or not detailed impacts have
              been computed in these NewsResults for this revision

        See also
        --------
        data_updates
        """
    data = pd.concat([self.revised_all.rename('revised'), self.revised_prev_all.rename('observed (prev)')], axis=1).sort_index()
    data['detailed impacts computed'] = self.revised_all.index.isin(self.revised.index)
    return data