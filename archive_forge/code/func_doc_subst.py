def doc_subst(snippets):
    """ Substitute format strings in class or function docstring """

    def decorator(cls):
        if cls.__doc__ is not None:
            stripped_snippets = {key: snippet.strip() for key, snippet in snippets.items()}
            cls.__doc__ = cls.__doc__.format(**stripped_snippets)
        return cls
    return decorator