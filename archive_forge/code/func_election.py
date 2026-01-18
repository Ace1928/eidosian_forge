def election():
    """
    Each row represents voting results for an electoral district in the 2013 Montreal
    mayoral election.

    Returns:
        A `pandas.DataFrame` with 58 rows and the following columns:
        `['district', 'Coderre', 'Bergeron', 'Joly', 'total', 'winner', 'result', 'district_id']`."""
    return _get_dataset('election')