from statsmodels.compat.pandas import FUTURE_STACK
import numpy as np
import pandas as pd
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
@property
def revision_details_by_update(self):
    """
        Details of forecast revisions from revisions, organized by updates

        Returns
        -------
        details : pd.DataFrame
            Index is as MultiIndex consisting of:

            - `revision date`: the date of the data revision, that results in
              `revision` that impacts the forecast of variables of interest
            - `revised variable`: the variable being revised, that results in
              `news` that impacts the forecast of variables of interest
            - `observed (prev)`: the previous value of the observation, as it
              was given in the previous dataset
            - `revised`: the value of the revised entry, as it is observed in
              the new dataset
            - `impact date`: the date of the impact on the variable of interest
            - `impacted variable`: the variable that is being impacted

            The columns are:

            - `revision`: the revision (this is `revised` - `observed (prev)`)
            - `weight`: the weight describing how the `revision` affects the
              forecast of the variable of interest
            - `impact`: the impact of the `revision` on the forecast of the
              variable of interest

        Notes
        -----
        This table decomposes updated forecasts of variables of interest from
        the `revision` associated with each revised datapoint from the new data
        release.

        This table does not summarize the impacts or show the effect of
        new datapoints, see `details_by_update` instead.

        Grouped impacts are shown in this table, with a "revision date" equal
        to the last period prior to which detailed revisions were computed and
        with "revised variable" set to the string "all prior revisions". For
        these rows, all columns except "impact" will be set to NaNs.

        This form of the details table is organized so that the revision
        dates / variables are first in the index, and in this table the index
        also contains the previously observed and revised values. This is
        convenient for displaying the entire table of detailed revisions
        because it allows sparsifying duplicate entries.

        However, since it includes previous observations and revisions in the
        index of the table, it is not convenient for subsetting by the variable
        of interest. Instead, the `revision_details_by_impact` property is
        organized to make slicing by impacted variables / dates easy. This
        allows, for example, viewing the details of data revisions on a
        particular variable or date of interest.

        See Also
        --------
        details_by_impact
        impacts
        """
    weights = self.revision_weights.stack(level=[0, 1], **FUTURE_STACK)
    df = pd.concat([self.revised_prev.rename('observed (prev)').reindex(weights.index), self.revised.reindex(weights.index), self.revisions.reindex(weights.index), weights.rename('weight'), (self.revisions.reindex(weights.index) * weights).rename('impact')], axis=1)
    if self.n_revisions_grouped > 0:
        df = pd.concat([df, self._revision_grouped_impacts])
        df.index = df.index.set_names(['revision date', 'revised variable', 'impact date', 'impacted variable'])
    details = df.set_index(['observed (prev)', 'revised'], append=True).reorder_levels(['revision date', 'revised variable', 'revised', 'observed (prev)', 'impact date', 'impacted variable']).sort_index()
    if self.impacted_variable is not None and len(df) > 0:
        details = details.loc[np.s_[:, :, :, :, :, self.impacted_variable], :]
    mask = np.abs(details['impact']) > self.tolerance
    return details[mask]