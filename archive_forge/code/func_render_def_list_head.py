import re
from ..util import strip_end
def render_def_list_head(renderer, text):
    return '<dt>' + text + '</dt>\n'