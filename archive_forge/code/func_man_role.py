from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def man_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    Convert a Linux manual topic to a hyperlink.

    Using the ``:man:`` role is very simple, here's an example:

    .. code-block:: rst

        See the :man:`python` documentation.

    This results in the following:

      See the :man:`python` documentation.

    As the example shows you can use the role inline, embedded in sentences of
    text. In the generated documentation the ``:man:`` text is omitted and a
    hyperlink pointing to the Debian Linux manual pages is emitted.
    """
    man_url = 'https://manpages.debian.org/%s' % text
    reference = docutils.nodes.reference(rawtext, docutils.utils.unescape(text), refuri=man_url, **options)
    return ([reference], [])