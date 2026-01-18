import re
from ..helpers import PREVENT_BACKSLASH
def render_strikethrough(renderer, text):
    return '<del>' + text + '</del>'