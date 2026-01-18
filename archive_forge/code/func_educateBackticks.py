from a single quote by the algorithm. Therefore, a text like::
import re, sys
def educateBackticks(text, language='en'):
    """
    Parameter:  String (unicode or bytes).
    Returns:    The `text`, with ``backticks'' -style double quotes
                translated into HTML curly quote entities.
    Example input:  ``Isn't this fun?''
    Example output: “Isn't this fun?“;
    """
    smart = smartchars(language)
    text = re.sub('``', smart.opquote, text)
    text = re.sub("''", smart.cpquote, text)
    return text