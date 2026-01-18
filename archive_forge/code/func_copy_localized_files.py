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
@progress_message(__('copying localized files'))
def copy_localized_files(self) -> None:
    source_dir = path.join(self.confdir, self.config.applehelp_locale + '.lproj')
    target_dir = self.outdir
    if path.isdir(source_dir):
        excluded = Matcher(self.config.exclude_patterns + ['**/.*'])
        copy_asset(source_dir, target_dir, excluded, context=self.globalcontext, renderer=self.templates)