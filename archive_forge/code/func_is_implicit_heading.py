from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.html5 import HTML5Translator as SphinxHTML5Translator
def is_implicit_heading(self, node):
    """Determines if a node is an implicit heading.

        An implicit heading is represented by a paragraph node whose only
        child is a strong node with text that isnt in `IGNORE_IMPLICIT_HEADINGS`.
        """
    return len(node) == 1 and isinstance(node[0], nodes.strong) and (len(node[0]) == 1) and isinstance(node[0][0], nodes.Text) and (node[0][0].astext() not in self.IGNORE_IMPLICIT_HEADINGS)