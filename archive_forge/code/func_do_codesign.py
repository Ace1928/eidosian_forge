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
@progress_message(__('signing help book'))
def do_codesign(self) -> None:
    """If we've been asked to, sign the bundle."""
    args = [self.config.applehelp_codesign_path, '-s', self.config.applehelp_codesign_identity, '-f']
    args += self.config.applehelp_codesign_flags
    args.append(self.bundle_path)
    if self.config.applehelp_disable_external_tools:
        raise SkipProgressMessage(__('you will need to sign this help book with:\n  %s'), ' '.join([shlex.quote(arg) for arg in args]))
    else:
        try:
            subprocess.run(args, stdout=PIPE, stderr=STDOUT, check=True)
        except OSError:
            raise AppleHelpCodeSigningFailed(__('Command not found: %s') % args[0])
        except CalledProcessError as exc:
            raise AppleHelpCodeSigningFailed(exc.stdout)