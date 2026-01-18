from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateSingleBackticks(text, language='en'):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with `backticks' -style single quotes
                translated into HTML curly quote entities.

    Example input:  `Isn't this fun?'
    Example output: ‘Isn’t this fun?’
    """
    smart = smartchars(language)
    text = re.sub('`', smart.osquote, text)
    text = re.sub("'", smart.csquote, text)
    return text