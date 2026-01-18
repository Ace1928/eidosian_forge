import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
def escape_script(s):
    """Escape a string that will be the content of an HTML script tag.

    We replace the opening bracket of <script, </script, and <!-- with the unicode
    equivalent. This is inspired by the documentation for the script tag at
    https://html.spec.whatwg.org/multipage/scripting.html#restrictions-for-contents-of-script-elements

    We only replace these three cases so that most html or other content
    involving `<` is readable.
    """
    return script_escape_re.sub('\\\\u003c\\1', s)