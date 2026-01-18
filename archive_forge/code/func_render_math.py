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
def render_math(self: HTMLTranslator, math: str) -> Tuple[Optional[str], Optional[int]]:
    """Render the LaTeX math expression *math* using latex and dvipng or
    dvisvgm.

    Return the filename relative to the built document and the "depth",
    that is, the distance of image bottom and baseline in pixels, if the
    option to use preview_latex is switched on.
    Also return the temporary and destination files.

    Error handling may seem strange, but follows a pattern: if LaTeX or dvipng
    (dvisvgm) aren't available, only a warning is generated (since that enables
    people on machines without these programs to at least build the rest of the
    docs successfully).  If the programs are there, however, they may not fail
    since that indicates a problem in the math source.
    """
    image_format = self.builder.config.imgmath_image_format.lower()
    if image_format not in SUPPORT_FORMAT:
        raise MathExtError('imgmath_image_format must be either "png" or "svg"')
    latex = generate_latex_macro(image_format, math, self.builder.config, self.builder.confdir)
    filename = f'{sha1(latex.encode()).hexdigest()}.{image_format}'
    generated_path = path.join(self.builder.outdir, self.builder.imagedir, 'math', filename)
    ensuredir(path.dirname(generated_path))
    if path.isfile(generated_path):
        if image_format == 'png':
            depth = read_png_depth(generated_path)
        elif image_format == 'svg':
            depth = read_svg_depth(generated_path)
        return (generated_path, depth)
    if hasattr(self.builder, '_imgmath_warned_latex') or hasattr(self.builder, '_imgmath_warned_image_translator'):
        return (None, None)
    try:
        dvipath = compile_math(latex, self.builder)
    except InvokeError:
        self.builder._imgmath_warned_latex = True
        return (None, None)
    try:
        if image_format == 'png':
            depth = convert_dvi_to_png(dvipath, self.builder, generated_path)
        elif image_format == 'svg':
            depth = convert_dvi_to_svg(dvipath, self.builder, generated_path)
    except InvokeError:
        self.builder._imgmath_warned_image_translator = True
        return (None, None)
    return (generated_path, depth)