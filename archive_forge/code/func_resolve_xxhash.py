from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_xxhash(required: bool=False):
    """
    Ensures that `xxhash` is available
    """
    global xxhash, _xxhash_available
    if not _xxhash_available:
        resolve_missing('xxhash', required=required)
        import xxhash
        _xxhash_available = True
        globals()['xxhash'] = xxhash