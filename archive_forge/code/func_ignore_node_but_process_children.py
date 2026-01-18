import re
import sys
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
def ignore_node_but_process_children(self, node):
    raise nodes.SkipDeparture