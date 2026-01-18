import re
from ..helpers import PREVENT_BACKSLASH
def render_table_row(renderer, text):
    return '<tr>\n' + text + '</tr>\n'