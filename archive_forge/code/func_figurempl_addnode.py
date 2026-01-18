from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib
def figurempl_addnode(app):
    app.add_node(figmplnode, html=(visit_figmpl_html, depart_figmpl_html), latex=(visit_figmpl_latex, depart_figmpl_latex))