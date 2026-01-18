from typing import Type
def failed_to_load_extension(exception):
    """Handle failing to load a binary extension.

    This should be called from the ImportError block guarding the attempt to
    import the native extension.  If this function returns, the pure-Python
    implementation should be loaded instead::

    >>> try:
    >>>     import _fictional_extension_pyx
    >>> except ImportError, e:
    >>>     failed_to_load_extension(e)
    >>>     import _fictional_extension_py
    """
    exception_str = str(exception)
    if exception_str not in _extension_load_failures:
        import warnings
        warnings.warn('failed to load compiled extension: %s' % exception_str, UserWarning)
        _extension_load_failures.append(exception_str)