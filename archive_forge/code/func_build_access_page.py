import plistlib
import shlex
import subprocess
from os import environ
from os import path
from subprocess import CalledProcessError, PIPE, STDOUT
from typing import Any
import sphinx
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.errors import SphinxError
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset, copy_asset_file
from sphinx.util.matching import Matcher
from sphinx.util.osutil import ensuredir, make_filename
@progress_message(__('building access page'))
def build_access_page(self, language_dir: str) -> None:
    """Build the access page."""
    context = {'toc': self.config.master_doc + self.out_suffix, 'title': self.config.applehelp_title}
    copy_asset_file(path.join(template_dir, '_access.html_t'), language_dir, context)