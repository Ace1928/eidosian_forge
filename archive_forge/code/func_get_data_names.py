from statsmodels.compat.python import lrange
def get_data_names(series_or_dataframe):
    """
    Input can be an array or pandas-like. Will handle 1d array-like but not
    2d. Returns a str for 1d data or a list of strings for 2d data.
    """
    names = getattr(series_or_dataframe, 'name', None)
    if not names:
        names = getattr(series_or_dataframe, 'columns', None)
    if not names:
        shape = getattr(series_or_dataframe, 'shape', [1])
        nvars = 1 if len(shape) == 1 else series_or_dataframe.shape[1]
        names = ['X%d' for _ in range(nvars)]
        if nvars == 1:
            names = names[0]
    else:
        names = names.tolist()
    return names