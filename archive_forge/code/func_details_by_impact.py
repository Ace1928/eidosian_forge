from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def details_by_impact(self):
    """
        Details of forecast revisions from news, organized by impacts first

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted
            - `update date`: the date of the data update, that results in
              `news` that impacts the forecast of variables of interest
            - `updated variable`: the variable being updated, that results in
              `news` that impacts the forecast of variables of interest

            The columns are:

            - `forecast (prev)`: the previous forecast of the new entry,
              based on the information available in the previous dataset
            - `observed`: the value of the new entry, as it is observed in the
              new dataset
            - `news`: the news associated with the update (this is just the
              forecast error: `observed` - `forecast (prev)`)
            - `weight`: the weight describing how the `news` effects the
              forecast of the variable of interest
            - `impact`: the impact of the `news` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `news` associated with each updated datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        revisions. That information can be found in the `impacts` or
        `revision_details_by_impact` tables.

        This form of the details table is organized so that the impacted
        dates / variables are first in the index. This is convenient for
        slicing by impacted variables / dates to view the details of data
        updates for a particular variable or date.

        However, since the `forecast (prev)` and `observed` columns have a lot
        of duplication, printing the entire table gives a result that is less
        easy to parse than that produced by the `details_by_update` property.
        `details_by_update` contains the same information but is organized to
        be more convenient for displaying the entire table of detailed updates.
        At the same time, `details_by_update` is less convenient for
        subsetting.

        See Also
        --------
        details_by_update
        revision_details_by_update
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
    df = df[['observed', 'forecast (prev)', 'news', 'weight', 'impact']]
    df = df.reorder_levels([2, 3, 0, 1]).sort_index()
    if self.impacted_variable is not None and len(df) > 0:
        df = df.loc[np.s_[:, self.impacted_variable], :]
    mask = np.abs(df['impact']) > self.tolerance
    return df[mask]