import sys
def getattr_str(obj, attr, fmt=None, fallback='?'):
    """Return string of the given object's attribute.

    Defaults to the given fallback value if attribute is not present.
    """
    try:
        value = getattr(obj, attr)
    except AttributeError:
        return fallback
    if fmt is None:
        return str(value)
    return fmt % value