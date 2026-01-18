import re
from ._base import DirectivePlugin
from ..util import escape as escape_text, escape_url
def render_figcaption(self, text):
    return '<figcaption>' + text + '</figcaption>\n'