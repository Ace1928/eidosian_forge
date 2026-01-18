from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_fileio(required: bool=False):
    """
    Ensures that `fileio` is available
    """
    global fileio, _fileio_available
    if not _fileio_available:
        resolve_missing('fileio', 'file-io', required=required)
        import fileio
        _fileio_available = True
        globals()['fileio'] = fileio