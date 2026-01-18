import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import jinja2
import markupsafe
from bs4 import BeautifulSoup
from jupyter_core.paths import jupyter_path
from traitlets import Bool, Unicode, default, validate
from traitlets.config import Config
from jinja2.loaders import split_template_path
from nbformat import NotebookNode
from nbconvert.filters.highlight import Highlight2HTML
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.filters.widgetsdatatypefilter import WidgetsDataTypeFilter
from nbconvert.utils.iso639_1 import iso639_1
from .templateexporter import TemplateExporter
def resources_include_css(name):
    env = self.environment
    code = '<style type="text/css">\n%s</style>' % env.loader.get_source(env, name)[0]
    return markupsafe.Markup(code)