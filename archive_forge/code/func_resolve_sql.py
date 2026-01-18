from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_sql(required: bool=True, require_sqlalchemy: bool=True, require_psycopg2: bool=False, require_asyncpg: bool=False, require_sqlalchemy_json: bool=True):
    """
    Ensures that `sqlalchemy` is available
    """
    if require_sqlalchemy:
        resolve_sqlalchemy(required=required)
    if require_psycopg2:
        resolve_psycopg2(required=required)
    if require_asyncpg:
        resolve_asyncpg(required=required)
    if require_sqlalchemy_json:
        resolve_sqlalchemy_json(required=required)