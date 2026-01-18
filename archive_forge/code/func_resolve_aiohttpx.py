from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_aiohttpx(required: bool=False):
    """
    Ensures that `aiohttpx` is available
    """
    global aiohttpx, _aiohttpx_available
    if not _aiohttpx_available:
        resolve_missing('aiohttpx', required=required)
        import aiohttpx
        _aiohttpx_available = True
        globals()['aiohttpx'] = aiohttpx