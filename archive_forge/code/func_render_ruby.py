import re
from ..util import unikey
from ..helpers import parse_link, parse_link_label
def render_ruby(renderer, text, rt):
    return '<ruby><rb>' + text + '</rb><rt>' + rt + '</rt></ruby>'