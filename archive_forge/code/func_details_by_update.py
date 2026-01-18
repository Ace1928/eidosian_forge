from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def details_by_update(self):
    """
        Details of forecast revisions from news, organized by updates first

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `update date`: the date of the data update, that results in
              `news` that impacts the forecast of variables of interest
            - `updated variable`: the variable being updated, that results in
              `news` that impacts the forecast of variables of interest
            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
            - `observed`: the value of the new entry, as it is observed in the
              new dataset
            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `news`: the news associated with the update (this is just the
              forecast error: `observed` - `forecast (prev)`)
            - `weight`: the weight describing how the `news` affects the
              forecast of the variable of interest
            - `impact`: the impact of the `news` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `news` associated with each updated datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        revisions. That information can be found in the `impacts` table.

        This form of the details table is organized so that the updated
        dates / variables are first in the index, and in this table the index
        also contains the forecasts and observed values of the updates. This is
        convenient for displaying the entire table of detailed updates because
        it allows sparsifying duplicate entries.

        However, since it includes forecasts and observed values in the index
        of the table, it is not convenient for subsetting by the variable of
        interest. Instead, the `details_by_impact` property is organized to
        make slicing by impacted variables / dates easy. This allows, for
        example, viewing the details of data updates on a particular variable
        or date of interest.

        See Also
        --------
        details_by_impact
        impacts
        """
    s = self.weights.stack(level=[0, 1], **FUTURE_STACK)
    df = s.rename('weight').to_frame()
    if len(self.updates_iloc):
        df['forecast (prev)'] = self.update_forecasts
        df['observed'] = self.update_realized
        df['news'] = self.news
        df['impact'] = df['news'] * df['weight']
    else:
        df['forecast (prev)'] = []
        df['observed'] = []
        df['news'] = []
        df['impact'] = []
    df = df[['forecast (prev)', 'observed', 'news', 'weight', 'impact']]
    df = df.reset_index()
    keys = ['update date', 'updated variable', 'observed', 'forecast (prev)', 'impact date', 'impacted variable']
    df.index = pd.MultiIndex.from_arrays([df[key] for key in keys])
    details = df.drop(keys, axis=1).sort_index()
    if self.impacted_variable is not None and len(df) > 0:
        details = details.loc[np.s_[:, :, :, :, :, self.impacted_variable], :]
    mask = np.abs(details['impact']) > self.tolerance
    return details[mask]