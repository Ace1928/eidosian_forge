import time
@property
def eval_df(self):
    """The dataframe with evaluation data.
        This has validation scores calculated at the end of each epoch.
        """
    return self._dataframes['eval']