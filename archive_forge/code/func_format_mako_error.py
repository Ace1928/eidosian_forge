from .compat import escape
from .jsonify import encode
def format_mako_error(exc_value):
    """
        Implements ``Mako`` renderer error formatting.
        """
    if isinstance(exc_value, (CompileException, SyntaxException)):
        return html_error_template().render(full=False, css=False)