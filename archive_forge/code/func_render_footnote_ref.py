import re
from ..core import BlockState
from ..util import unikey
from ..helpers import LINK_LABEL
def render_footnote_ref(renderer, key: str, index: int):
    i = str(index)
    html = '<sup class="footnote-ref" id="fnref-' + i + '">'
    return html + '<a href="#fn-' + i + '">' + i + '</a></sup>'