from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_aiocache(required: bool=True):
    """
    Ensures that `aiocache` is availableable
    """
    global _aiocache_available
    global aiocache
    if not _aiocache_available:
        resolve_missing('aiocache', required=required)
        import aiocache
        _aiocache_available = True
        globals()['aiocache'] = aiocache