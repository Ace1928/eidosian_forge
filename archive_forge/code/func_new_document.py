import sys
from docutils import utils, parsers, Component
from docutils.transforms import universal
def new_document(self):
    """Create and return a new empty document tree (root node)."""
    document = utils.new_document(self.source.source_path, self.settings)
    return document