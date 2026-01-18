from builtins import abs as _abs
def imatmul(a, b):
    """Same as a @= b."""
    a @= b
    return a