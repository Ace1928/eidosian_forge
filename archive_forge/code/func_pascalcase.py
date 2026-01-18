import re
def pascalcase(string):
    """Convert string into pascal case.

    Args:
        string: String to convert.

    Returns:
        string: Pascal case string.

    """
    return capitalcase(camelcase(string))