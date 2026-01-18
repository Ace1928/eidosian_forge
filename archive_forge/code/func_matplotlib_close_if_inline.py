from qiskit.utils import optionals as _optionals
def matplotlib_close_if_inline(figure):
    """Close the given matplotlib figure if the backend in use draws figures inline.

    If the backend does not draw figures inline, this does nothing.  This function is to prevent
    duplicate images appearing; the inline backends will capture the figure in preparation and
    display it as well, whereas the drawers want to return the figure to be displayed."""
    import matplotlib.pyplot
    if matplotlib.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
        matplotlib.pyplot.close(figure)