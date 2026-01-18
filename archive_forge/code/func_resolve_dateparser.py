def resolve_dateparser(required: bool=False):
    """
    Ensures that `dateparser` is available
    """
    global dateparser, pytz, _dateparser_available
    if not _dateparser_available:
        from lazyops.utils.imports import resolve_missing
        resolve_missing('dateparser', required=required)
        import dateparser
        import pytz
        _dateparser_available = True
        globals()['dateparser'] = dateparser
        globals()['pytz'] = pytz