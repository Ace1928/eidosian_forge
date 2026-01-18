def atomic_number(name):
    """Provide atomic number for a given element

    Parameters
    ----------
    name: str
        Full name or chemical symbol of an element

    Returns
    -------
    int
        Atomic number
    """
    try:
        return symbols.index(name.capitalize()) + 1
    except ValueError:
        return lower_names.index(name.lower()) + 1