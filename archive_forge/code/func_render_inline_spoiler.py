import re
def render_inline_spoiler(renderer, text):
    return '<span class="spoiler">' + text + '</span>'