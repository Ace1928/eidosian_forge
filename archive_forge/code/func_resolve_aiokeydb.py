from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_aiokeydb(required: bool=False, version: str=_min_version):
    """
    Ensures that `aiokeydb` is available
    """
    global aiokeydb, _aiokeydb_available
    if not _aiokeydb_available:
        resolve_missing(f'aiokeydb=={version}', required=required)
        import aiokeydb
        _aiokeydb_available = True
        globals()['aiokeydb'] = aiokeydb