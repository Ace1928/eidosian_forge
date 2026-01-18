from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_pycryptodome(required: bool=False):
    """
    Ensures that `pycryptodome` is available
    """
    global Crypto, _pycryptodome_available
    if not _pycryptodome_available:
        resolve_missing('pycryptodome', required=required)
        import Crypto
        _pycryptodome_available = True
        globals()['Crypto'] = Crypto