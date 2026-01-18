import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def authors_from_paragraphs(self, field):
    for item in field[1]:
        if not isinstance(item, (nodes.paragraph, nodes.comment)):
            raise TransformError
    authors = [item.children for item in field[1] if not isinstance(item, nodes.comment)]
    return authors