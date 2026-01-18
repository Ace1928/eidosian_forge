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
def find_lab_theme(theme_name):
    """
    Find a JupyterLab theme location by name.

    Parameters
    ----------
    theme_name : str
        The name of the labextension theme you want to find.

    Raises
    ------
    ValueError
        If the theme was not found, or if it was not specific enough.

    Returns
    -------
    theme_name: str
        Full theme name (with scope, if any)
    labextension_path : Path
        The path to the found labextension on the system.
    """
    paths = jupyter_path('labextensions')
    matching_themes = []
    theme_path = None
    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            if 'package.json' in filenames and 'themes' in dirnames:
                with open(Path(dirpath) / 'package.json', encoding='utf-8') as fobj:
                    labext_name = json.loads(fobj.read())['name']
                if labext_name == theme_name or theme_name in labext_name.split('/'):
                    matching_themes.append(labext_name)
                    full_theme_name = labext_name
                    theme_path = Path(dirpath) / 'themes' / labext_name
    if len(matching_themes) == 0:
        msg = f'Could not find lab theme "{theme_name}"'
        raise ValueError(msg)
    if len(matching_themes) > 1:
        msg = f'Found multiple themes matching "{theme_name}": {matching_themes}. Please be more specific about which theme you want to use.'
        raise ValueError(msg)
    return (full_theme_name, theme_path)