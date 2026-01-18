import numpy as np
def pivot_dataframe(dataframe):
    """Gets a pivoted wide-form pandas dataframe.

    The wide-form DataFrame has all its tags included as columns of the
    DataFrame, which is more convenient to work. If the condition of having
    uniform sets of step values across all tags in all runs is not met,
    this will error.

    Args:
      dataframe: pandas dataframe to pivot.

    Returns:
      Pivoted wide-form pandas dataframe.
    Raises:
      ValueError if step values across all tags are not uniform.
    """
    num_missing_0 = np.count_nonzero(dataframe.isnull().values)
    dataframe = dataframe.pivot_table(values=['value', 'wall_time'] if 'wall_time' in dataframe.columns else 'value', index=['run', 'step'], columns='tag', dropna=False)
    num_missing_1 = np.count_nonzero(dataframe.isnull().values)
    if num_missing_1 > num_missing_0:
        raise ValueError('pivoted DataFrame contains missing value(s). This is likely due to two timeseries having different sets of steps in your experiment. You can avoid this error by calling `get_scalars()` with `pivot=False` to disable the DataFrame pivoting.')
    dataframe = dataframe.reset_index()
    dataframe.columns.name = None
    dataframe.columns.names = [None for name in dataframe.columns.names]
    return dataframe