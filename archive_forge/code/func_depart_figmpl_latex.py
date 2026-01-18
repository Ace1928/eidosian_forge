from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib
def depart_figmpl_latex(self, node):
    self.depart_figure(node)