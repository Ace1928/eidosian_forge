from __future__ import annotations
import logging  # isort:skip
from html import escape
from os.path import join
from sphinx.errors import SphinxError
from sphinx.util.display import status_iterator
from . import PARALLEL_SAFE
def html_page_context(app, pagename, templatename, context, doctree):
    """Collect page names for the sitemap as HTML pages are built."""
    site = context['SITEMAP_BASE_URL']
    version = context['version']
    app.sitemap_links.add(f'{site}{version}/{pagename}.html')