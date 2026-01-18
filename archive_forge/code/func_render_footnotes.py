import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def render_footnotes(renderer, text: str):
    return '<section class="footnotes">\n<ol>\n' + text + '</ol>\n</section>\n'