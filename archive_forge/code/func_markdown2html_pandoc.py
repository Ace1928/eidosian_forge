import re
from .pandoc import convert_pandoc
def markdown2html_pandoc(source, extra_args=None):
    """
    Convert a markdown string to HTML via pandoc.
    """
    extra_args = extra_args or ['--mathjax']
    return convert_pandoc(source, 'markdown', 'html', extra_args=extra_args)