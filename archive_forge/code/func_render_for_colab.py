import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def render_for_colab(html_str, iframe_height):
    from IPython.display import HTML, Javascript
    js_height_snippet = f'google.colab.output.setIframeHeight({iframe_height}, true, {{minHeight: {iframe_height}}})'
    display(Javascript(js_height_snippet))
    display(HTML(html_str))