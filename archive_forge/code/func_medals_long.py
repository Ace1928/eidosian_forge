def medals_long(indexed=False):
    """
    This dataset represents the medal table for Olympic Short Track Speed Skating for the
    top three nations as of 2020.

    Returns:
        A `pandas.DataFrame` with 9 rows and the following columns:
        `['nation', 'medal', 'count']`.
        If `indexed` is True, the 'nation' column is used as the index."""
    df = _get_dataset('medals').melt(id_vars=['nation'], value_name='count', var_name='medal')
    if indexed:
        df = df.set_index('nation')
    return df