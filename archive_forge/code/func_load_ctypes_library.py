from ctypes import CDLL
from ctypes.util import find_library
def load_ctypes_library(name, signatures, error_checkers):
    """
    Load library ``name`` and return a :class:`ctypes.CDLL` object for it.

    :param str name: the library name
    :param signatures: signatures of methods
    :type signatures: dict of str * (tuple of (list of type) * type)
    :param error_checkers: error checkers for methods
    :type error_checkers: dict of str * ((int * ptr * arglist) -> int)

    The library has errno handling enabled.
    Important functions are given proper signatures and return types to support
    type checking and argument conversion.

    :returns: a loaded library
    :rtype: ctypes.CDLL
    :raises ImportError: if the library is not found
    """
    library_name = find_library(name)
    if not library_name:
        raise ImportError('No library named %s' % name)
    lib = CDLL(library_name, use_errno=True)
    for funcname, signature in signatures.items():
        function = getattr(lib, funcname, None)
        if function:
            argtypes, restype = signature
            function.argtypes = argtypes
            function.restype = restype
            errorchecker = error_checkers.get(funcname)
            if errorchecker:
                function.errcheck = errorchecker
    return lib