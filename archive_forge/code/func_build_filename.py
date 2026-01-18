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
def build_filename(self):
    filename = '{dirname}/figure_{render_count}.html'.format(dirname=self.html_directory, render_count=CoCalcRenderer._render_count)
    CoCalcRenderer._render_count += 1
    return filename