from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_aiofiles(required: bool=False):
    """
    Ensures that `aiofiles` is available
    """
    global aiofiles, _aiofiles_available
    if not _aiofiles_available:
        resolve_missing('aiofiles', required=required)
        import aiofiles
        _aiofiles_available = True
        globals()['aiofiles'] = aiofiles