from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_psycopg2(required: bool=False):
    """
    Ensures that `psycopg2` is available
    """
    global psycopg2, _psycopg2_available
    if not _psycopg2_available:
        resolve_missing('psycopg2', 'psycopg2-binary', required=required)
        import psycopg2
        _psycopg2_available = True
        globals()['psycopg2'] = psycopg2