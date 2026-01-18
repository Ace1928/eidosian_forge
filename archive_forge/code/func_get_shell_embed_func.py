from functools import wraps
def get_shell_embed_func(shells=None, known_shells=None):
    """Return the first acceptable shell-embed function
    from a given list of shell names.
    """
    if shells is None:
        shells = DEFAULT_PYTHON_SHELLS.keys()
    if known_shells is None:
        known_shells = DEFAULT_PYTHON_SHELLS.copy()
    for shell in shells:
        if shell in known_shells:
            try:
                return known_shells[shell]()
            except ImportError:
                continue