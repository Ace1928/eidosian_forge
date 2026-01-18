from numba.core import types, typing
def is_signature(sig):
    """
    Return whether *sig* is a potentially valid signature
    specification (for user-facing APIs).
    """
    return isinstance(sig, (str, tuple, typing.Signature))