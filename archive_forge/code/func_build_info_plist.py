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
@progress_message(__('writing Info.plist'))
def build_info_plist(self, contents_dir: str) -> None:
    """Construct the Info.plist file."""
    info_plist = {'CFBundleDevelopmentRegion': self.config.applehelp_dev_region, 'CFBundleIdentifier': self.config.applehelp_bundle_id, 'CFBundleInfoDictionaryVersion': '6.0', 'CFBundlePackageType': 'BNDL', 'CFBundleShortVersionString': self.config.release, 'CFBundleSignature': 'hbwr', 'CFBundleVersion': self.config.applehelp_bundle_version, 'HPDBookAccessPath': '_access.html', 'HPDBookIndexPath': 'search.helpindex', 'HPDBookTitle': self.config.applehelp_title, 'HPDBookType': '3', 'HPDBookUsesExternalViewer': False}
    if self.config.applehelp_icon is not None:
        info_plist['HPDBookIconPath'] = path.basename(self.config.applehelp_icon)
    if self.config.applehelp_kb_url is not None:
        info_plist['HPDBookKBProduct'] = self.config.applehelp_kb_product
        info_plist['HPDBookKBURL'] = self.config.applehelp_kb_url
    if self.config.applehelp_remote_url is not None:
        info_plist['HPDBookRemoteURL'] = self.config.applehelp_remote_url
    with open(path.join(contents_dir, 'Info.plist'), 'wb') as f:
        plistlib.dump(info_plist, f)