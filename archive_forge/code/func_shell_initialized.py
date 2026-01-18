@_define_event
def shell_initialized(ip):
    """Fires after initialisation of :class:`~IPython.core.interactiveshell.InteractiveShell`.

    This is before extensions and startup scripts are loaded, so it can only be
    set by subclassing.

    Parameters
    ----------
    ip : :class:`~IPython.core.interactiveshell.InteractiveShell`
        The newly initialised shell.
    """
    pass