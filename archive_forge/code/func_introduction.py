def introduction(request):
    """ Return a configuration for contributing examples to the
    Demo application.

    Parameters
    ----------
    request : dict
        Information provided by the demo application.
        Currently this is a placeholder.

    Returns
    -------
    response : dict
    """
    import pkg_resources
    return dict(version=1, name='Traits Introduction', root=pkg_resources.resource_filename('traits', 'examples/introduction'))