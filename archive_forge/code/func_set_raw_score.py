def set_raw_score(atoms, raw_score):
    """Set the raw_score of an atoms object in the
    atoms.info['key_value_pairs'] dictionary.
    
    Parameters
    ----------
    atoms : Atoms object
        The atoms object that corresponds to this raw_score
    raw_score : float or int
        Independent calculation of how fit the candidate is.
    """
    if 'key_value_pairs' not in atoms.info:
        atoms.info['key_value_pairs'] = {}
    atoms.info['key_value_pairs']['raw_score'] = raw_score