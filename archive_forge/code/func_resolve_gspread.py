from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_gspread(required: bool=False):
    """
    Ensures that `gspread` is available
    """
    global gspread, _gspread_available
    if not _gspread_available:
        resolve_missing('gspread', required=required)
        import gspread
        _gspread_available = True
        globals()['gspread'] = gspread