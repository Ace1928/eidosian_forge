import re
from ..helpers import PREVENT_BACKSLASH
def render_table_cell(renderer, text, align=None, head=False):
    if head:
        tag = 'th'
    else:
        tag = 'td'
    html = '  <' + tag
    if align:
        html += ' style="text-align:' + align + '"'
    return html + '>' + text + '</' + tag + '>\n'