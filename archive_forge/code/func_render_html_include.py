import os
from ._base import DirectivePlugin
def render_html_include(renderer, text, **attrs):
    return '<pre class="directive-include">\n' + text + '</pre>\n'