def get_raw_score(atoms):
    """Gets the raw_score of the supplied atoms object.

    Parameters
    ----------
    atoms : Atoms object
        The atoms object from which the raw_score will be returned.

    Returns
    -------
    raw_score : float or int
        The raw_score set previously.
    """
    return atoms.info['key_value_pairs']['raw_score']