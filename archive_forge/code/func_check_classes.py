import re
import sys
import time
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
from docutils.utils import smartquotes
def check_classes(self, node):
    if isinstance(node, nodes.Element):
        for class_value in node['classes'][:]:
            if class_value in self.strip_classes:
                node['classes'].remove(class_value)
            if class_value in self.strip_elements:
                return 1