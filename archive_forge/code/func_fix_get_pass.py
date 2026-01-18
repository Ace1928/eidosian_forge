import sys
def fix_get_pass():
    try:
        import getpass
    except ImportError:
        return
    import warnings
    fallback = getattr(getpass, 'fallback_getpass', None)
    if not fallback:
        fallback = getpass.default_getpass
    getpass.getpass = fallback
    if hasattr(getpass, 'GetPassWarning'):
        warnings.simplefilter('ignore', category=getpass.GetPassWarning)