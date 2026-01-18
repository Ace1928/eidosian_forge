import html
import os
from os.path import realpath, join, dirname
import sys
import time
import warnings
import webbrowser
import jinja2
from ..frontend_semver import DECKGL_SEMVER
def render_json_to_html(json_input, mapbox_key=None, google_maps_key=None, tooltip=True, css_background_color=None, custom_libraries=None, configuration=None, offline=False):
    js = j2_env.get_template('index.j2')
    css = j2_env.get_template('style.j2')
    css_text = css.render(css_background_color=css_background_color)
    html_str = js.render(mapbox_key=mapbox_key, google_maps_key=google_maps_key, json_input=json_input, deckgl_jupyter_widget_bundle=cdn_picker(offline=offline), tooltip=convert_js_bool(tooltip), css_text=css_text, custom_libraries=custom_libraries, configuration=configuration)
    return html_str