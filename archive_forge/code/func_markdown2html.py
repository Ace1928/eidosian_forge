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
@contextfilter
def markdown2html(self, context, source):
    """Markdown to HTML filter respecting the anchor_link_text setting"""
    cell = context.get('cell', {})
    attachments = cell.get('attachments', {})
    path = context.get('resources', {}).get('metadata', {}).get('path', '')
    renderer = IPythonRenderer(escape=False, attachments=attachments, embed_images=self.embed_images, path=path, anchor_link_text=self.anchor_link_text, exclude_anchor_links=self.exclude_anchor_links)
    return MarkdownWithMath(renderer=renderer).render(source)