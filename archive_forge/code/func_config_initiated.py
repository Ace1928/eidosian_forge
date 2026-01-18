from os import path
from sys import version_info as python_version
from sphinx import version_info as sphinx_version
from sphinx.locale import _
from sphinx.util.logging import getLogger
def config_initiated(app, config):
    theme_options = config.html_theme_options or {}
    if theme_options.get('canonical_url'):
        logger.warning(_('The canonical_url option is deprecated, use the html_baseurl option from Sphinx instead.'))