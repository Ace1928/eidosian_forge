import re
from ..helpers import PREVENT_BACKSLASH
def render_subscript(renderer, text):
    return '<sub>' + text + '</sub>'