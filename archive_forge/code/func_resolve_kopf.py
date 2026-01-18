from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_kopf(required: bool=True):
    """
    Ensures that `kopf`
    """
    global _kopf_available
    global kopf
    if not _kopf_available:
        resolve_missing('kopf', required=required)
        import kopf
        _kopf_available = True
        globals()['kopf'] = kopf