from docutils import nodes, utils
from docutils.parsers.rst import directives
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils.code_analyzer import Lexer, LexerError
def set_classes(options):
    """
    Auxiliary function to set options['classes'] and delete
    options['class'].
    """
    if 'class' in options:
        assert 'classes' not in options
        options['classes'] = options['class']
        del options['class']