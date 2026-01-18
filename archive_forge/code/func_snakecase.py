import re
def snakecase(string):
    """Convert string into snake case.
    Join punctuation with underscore

    Args:
        string: String to convert.

    Returns:
        string: Snake cased string.

    """
    string = re.sub('[\\-\\.\\s]', '_', str(string))
    if not string:
        return string
    return uplowcase(string[0], 'low') + re.sub('[A-Z0-9]', lambda matched: '_' + uplowcase(matched.group(0), 'low'), string[1:])