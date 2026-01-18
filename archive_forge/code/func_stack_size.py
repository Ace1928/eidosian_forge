def stack_size(size=None):
    """Dummy implementation of _thread.stack_size()."""
    if size is not None:
        raise error('setting thread stack size not supported')
    return 0