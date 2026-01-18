import re
import sys
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def copy_and_filter(self, node):
    """Return a copy of a title, with references, images, etc. removed."""
    visitor = ContentsFilter(self.document)
    node.walkabout(visitor)
    return visitor.get_entry_text()