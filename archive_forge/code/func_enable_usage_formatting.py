from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def enable_usage_formatting(app):
    """
    Reformat human friendly usage messages to reStructuredText_.

    :param app: The Sphinx application object (as given to ``setup()``).

    This function connects the :func:`usage_message_callback()` function to
    ``autodoc-process-docstring`` events.

    .. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
    """
    app.connect('autodoc-process-docstring', usage_message_callback)