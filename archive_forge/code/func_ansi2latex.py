import re
import markupsafe
def ansi2latex(text):
    """
    Convert ANSI colors to LaTeX colors.

    Parameters
    ----------
    text : unicode
        Text containing ANSI colors to convert to LaTeX

    """
    return _ansi2anything(text, _latexconverter)