import re
from ..util import strip_end
def render_def_list(renderer, text):
    return '<dl>\n' + text + '</dl>\n'