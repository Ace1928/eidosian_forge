import re
from ..helpers import PREVENT_BACKSLASH
def render_superscript(renderer, text):
    return '<sup>' + text + '</sup>'