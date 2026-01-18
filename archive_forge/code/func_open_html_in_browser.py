import base64
import json
import webbrowser
import inspect
import os
from os.path import isdir
from plotly import utils, optional_imports
from plotly.io import to_json, to_image, write_image, write_html
from plotly.io._orca import ensure_server
from plotly.io._utils import plotly_cdn_url
from plotly.offline.offline import _get_jconfig, get_plotlyjs
from plotly.tools import return_figure_from_figure_or_data
def open_html_in_browser(html, using=None, new=0, autoraise=True):
    """
    Display html in a web browser without creating a temp file.

    Instantiates a trivial http server and uses the webbrowser module to
    open a URL to retrieve html from that server.

    Parameters
    ----------
    html: str
        HTML string to display
    using, new, autoraise:
        See docstrings in webbrowser.get and webbrowser.open
    """
    if isinstance(html, str):
        html = html.encode('utf8')
    browser = None
    if using is None:
        browser = webbrowser.get(None)
    else:
        if not isinstance(using, tuple):
            using = (using,)
        for browser_key in using:
            try:
                browser = webbrowser.get(browser_key)
                if browser is not None:
                    break
            except webbrowser.Error:
                pass
        if browser is None:
            raise ValueError("Can't locate a browser with key in " + str(using))

    class OneShotRequestHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            bufferSize = 1024 * 1024
            for i in range(0, len(html), bufferSize):
                self.wfile.write(html[i:i + bufferSize])

        def log_message(self, format, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), OneShotRequestHandler)
    browser.open('http://127.0.0.1:%s' % server.server_port, new=new, autoraise=autoraise)
    server.handle_request()