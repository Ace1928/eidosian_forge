import re
from ..helpers import PREVENT_BACKSLASH
def render_mark(renderer, text):
    return '<mark>' + text + '</mark>'