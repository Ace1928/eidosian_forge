import re
def spinalcase(string):
    """Convert string into spinal case.
    Join punctuation with hyphen.

    Args:
        string: String to convert.

    Returns:
        string: Spinal cased string.

    """
    return re.sub('_', '-', snakecase(string))