import base64
import re
import shutil
import subprocess
import tempfile
from os import path
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging, sha1
from sphinx.util.math import get_node_equation_number, wrap_displaymath
from sphinx.util.osutil import ensuredir
from sphinx.util.png import read_png_depth, write_png_depth
from sphinx.util.template import LaTeXRenderer
from sphinx.writers.html import HTMLTranslator
def convert_dvi_to_image(command: List[str], name: str) -> Tuple[str, str]:
    """Convert DVI file to specific image format."""
    try:
        ret = subprocess.run(command, stdout=PIPE, stderr=PIPE, check=True, encoding='ascii')
        return (ret.stdout, ret.stderr)
    except OSError as exc:
        logger.warning(__('%s command %r cannot be run (needed for math display), check the imgmath_%s setting'), name, command[0], name)
        raise InvokeError from exc
    except CalledProcessError as exc:
        raise MathExtError('%s exited with error' % name, exc.stderr, exc.stdout) from exc