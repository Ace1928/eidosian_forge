import os.path
import docutils
from docutils import frontend, nodes, writers, io
from docutils.transforms import writer_aux
from docutils.writers import _html_base
def set_first_last(self, node):
    self.set_class_on_child(node, 'first', 0)
    self.set_class_on_child(node, 'last', -1)