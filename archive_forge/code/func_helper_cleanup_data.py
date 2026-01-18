from __future__ import absolute_import, division, print_function
def helper_cleanup_data(obj):
    """
    Removes the None values from the object and returns the object
    Args:
        obj: object to cleanup

    Returns:
       object: cleaned object
    """
    if isinstance(obj, (list, tuple, set)):
        return type(obj)((helper_cleanup_data(x) for x in obj if x is not None))
    elif isinstance(obj, dict):
        return type(obj)(((helper_cleanup_data(k), helper_cleanup_data(v)) for k, v in obj.items() if k is not None and v is not None))
    else:
        return obj