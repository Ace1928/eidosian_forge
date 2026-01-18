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
def convert_dvi_to_png(dvipath: str, builder: Builder, out_path: str) -> Optional[int]:
    """Convert DVI file to PNG image."""
    name = 'dvipng'
    command = [builder.config.imgmath_dvipng, '-o', out_path, '-T', 'tight', '-z9']
    command.extend(builder.config.imgmath_dvipng_args)
    if builder.config.imgmath_use_preview:
        command.append('--depth')
    command.append(dvipath)
    stdout, stderr = convert_dvi_to_image(command, name)
    depth = None
    if builder.config.imgmath_use_preview:
        for line in stdout.splitlines():
            matched = depth_re.match(line)
            if matched:
                depth = int(matched.group(1))
                write_png_depth(out_path, depth)
                break
    return depth