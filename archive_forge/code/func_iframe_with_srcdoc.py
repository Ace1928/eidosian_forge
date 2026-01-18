import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def iframe_with_srcdoc(html_str, width='100%', height=500):
    if isinstance(width, str):
        width = f'"{width}"'
    srcdoc = html.escape(html_str)
    iframe = f'\n        <iframe\n            width={width}\n            height={height}\n            frameborder="0"\n            srcdoc="{srcdoc}"\n        ></iframe>\n    '
    from IPython.display import HTML
    with warnings.catch_warnings():
        msg = 'Consider using IPython.display.IFrame instead'
        warnings.filterwarnings('ignore', message=msg)
        return HTML(iframe)