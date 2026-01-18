from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def revision_details_by_impact(self):
    """
        Details of forecast revisions from revised data, organized by impacts

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted
            - `revision date`: the date of the data revision, that results in
              `revision` that impacts the forecast of variables of interest
            - `revised variable`: the variable being revised, that results in
              `news` that impacts the forecast of variables of interest

            The columns are:

            - `observed (prev)`: the previous value of the observation, as it
              was given in the previous dataset
            - `revised`: the value of the revised entry, as it is observed in
              the new dataset
            - `revision`: the revision (this is `revised` - `observed (prev)`)
            - `weight`: the weight describing how the `revision` effects the
              forecast of the variable of interest
            - `impact`: the impact of the `revision` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `revision` associated with each revised datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        new datapoints. That information can be found in the
        `impacts` or `details_by_impact` tables.

        Grouped impacts are shown in this table, with a "revision date" equal
        to the last period prior to which detailed revisions were computed and
        with "revised variable" set to the string "all prior revisions". For
        these rows, all columns except "impact" will be set to NaNs.

        This form of the details table is organized so that the impacted
        dates / variables are first in the index. This is convenient for
        slicing by impacted variables / dates to view the details of data
        updates for a particular variable or date.

        However, since the `observed (prev)` and `revised` columns have a lot
        of duplication, printing the entire table gives a result that is less
        easy to parse than that produced by the `details_by_revision` property.
        `details_by_revision` contains the same information but is organized to
        be more convenient for displaying the entire table of detailed
        revisions. At the same time, `details_by_revision` is less convenient
        for subsetting.

        See Also
        --------
        details_by_revision
        details_by_impact
        impacts
        """
    weights = self.revision_weights.stack(level=[0, 1], **FUTURE_STACK)
    df = pd.concat([self.revised.reindex(weights.index), self.revised_prev.rename('observed (prev)').reindex(weights.index), self.revisions.reindex(weights.index), weights.rename('weight'), (self.revisions.reindex(weights.index) * weights).rename('impact')], axis=1)
    if self.n_revisions_grouped > 0:
        df = pd.concat([df, self._revision_grouped_impacts])
        df.index = df.index.set_names(['revision date', 'revised variable', 'impact date', 'impacted variable'])
    df = df.reorder_levels([2, 3, 0, 1]).sort_index()
    if self.impacted_variable is not None and len(df) > 0:
        df = df.loc[np.s_[:, self.impacted_variable], :]
    mask = np.abs(df['impact']) > self.tolerance
    return df[mask]