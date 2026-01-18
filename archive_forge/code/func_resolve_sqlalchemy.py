from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_sqlalchemy(required: bool=False):
    """
    Ensures that `sqlalchemy` is available
    """
    global sqlalchemy, _sqlalchemy_available
    if not _sqlalchemy_available:
        resolve_missing('sqlalchemy', required=required)
        import sqlalchemy
        _sqlalchemy_available = True
        globals()['sqlalchemy'] = sqlalchemy