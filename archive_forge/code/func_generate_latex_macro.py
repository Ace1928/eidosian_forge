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
def generate_latex_macro(image_format: str, math: str, config: Config, confdir: str='') -> str:
    """Generate LaTeX macro."""
    variables = {'fontsize': config.imgmath_font_size, 'baselineskip': int(round(config.imgmath_font_size * 1.2)), 'preamble': config.imgmath_latex_preamble, 'tightpage': '' if image_format == 'png' else ',tightpage', 'math': math}
    if config.imgmath_use_preview:
        template_name = 'preview.tex_t'
    else:
        template_name = 'template.tex_t'
    for template_dir in config.templates_path:
        template = path.join(confdir, template_dir, template_name)
        if path.exists(template):
            return LaTeXRenderer().render(template, variables)
    return LaTeXRenderer(templates_path).render(template_name, variables)