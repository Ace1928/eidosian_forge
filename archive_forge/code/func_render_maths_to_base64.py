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
def render_maths_to_base64(image_format: str, generated_path: Optional[str]) -> str:
    with open(generated_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode(encoding='utf-8')
    if image_format == 'png':
        return f'data:image/png;base64,{encoded}'
    if image_format == 'svg':
        return f'data:image/svg+xml;base64,{encoded}'
    raise MathExtError('imgmath_image_format must be either "png" or "svg"')