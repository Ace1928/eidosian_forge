from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_magic(required: bool=False):
    """
    Ensures that `magic` is available
    """
    global magic, _magic_available
    if not _magic_available:
        resolve_missing('magic', 'python-magic', required=required)
        import magic
        _magic_available = True
        globals()['magic'] = magic