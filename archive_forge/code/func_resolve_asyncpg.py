from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_asyncpg(required: bool=False):
    """
    Ensures that `asyncpg` is available
    """
    global asyncpg, _asyncpg_available
    if not _asyncpg_available:
        resolve_missing('asyncpg', required=required)
        import asyncpg
        _asyncpg_available = True
        globals()['asyncpg'] = asyncpg