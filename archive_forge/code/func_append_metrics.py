import time
def append_metrics(self, metrics, df_name):
    """Append new metrics to selected dataframes.

        Parameters
        ----------
        metrics : metric.EvalMetric
            New metrics to be added.
        df_name : str
            Name of the dataframe to be modified.
        """
    dataframe = self._dataframes[df_name]
    _add_new_columns(dataframe, metrics)
    dataframe.loc[len(dataframe)] = metrics