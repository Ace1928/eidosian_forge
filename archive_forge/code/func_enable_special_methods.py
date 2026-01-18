from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def enable_special_methods(app):
    """
    Enable documenting "special methods" using the autodoc_ extension.

    :param app: The Sphinx application object.

    This function connects the :func:`special_methods_callback()` function to
    ``autodoc-skip-member`` events.

    .. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
    """
    app.connect('autodoc-skip-member', special_methods_callback)