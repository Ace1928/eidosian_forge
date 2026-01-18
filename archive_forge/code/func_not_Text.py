import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
def not_Text(self, node):
    return not isinstance(node, nodes.Text)