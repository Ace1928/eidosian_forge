from os import path
from typing import Any, Dict, cast
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.domains.math import MathDomain
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError
from sphinx.locale import get_translation
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html import HTMLTranslator
from sphinxcontrib.jsmath.version import __version__
def html_visit_math(self: HTMLTranslator, node: nodes.math) -> None:
    self.body.append(self.starttag(node, 'span', '', CLASS='math notranslate nohighlight'))
    self.body.append(self.encode(node.astext()) + '</span>')
    raise nodes.SkipNode