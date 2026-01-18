from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_transformers(required: bool=False, version: str=None):
    """
    Ensures that `transformers` is available
    """
    global transformers, _transformers_available
    if not _transformers_available:
        pkg = 'transformers'
        if version is not None:
            pkg += f'=={version}'
        resolve_missing(pkg, required=required)
        import transformers
        _transformers_available = True
        globals()['transformers'] = transformers