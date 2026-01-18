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
def compile_math(latex: str, builder: Builder) -> str:
    """Compile LaTeX macros for math to DVI."""
    tempdir = ensure_tempdir(builder)
    filename = path.join(tempdir, 'math.tex')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(latex)
    command = [builder.config.imgmath_latex, '--interaction=nonstopmode']
    command.extend(builder.config.imgmath_latex_args)
    command.append('math.tex')
    try:
        subprocess.run(command, stdout=PIPE, stderr=PIPE, cwd=tempdir, check=True, encoding='ascii')
        return path.join(tempdir, 'math.dvi')
    except OSError as exc:
        logger.warning(__('LaTeX command %r cannot be run (needed for math display), check the imgmath_latex setting'), builder.config.imgmath_latex)
        raise InvokeError from exc
    except CalledProcessError as exc:
        raise MathExtError('latex exited with error', exc.stderr, exc.stdout) from exc