from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def iterate_all_nodes(self, node=None):
    """Generator to iterate over all nodes from `node` and down whether
        expanded or not. If `node` is `None`, the generator start with
        :attr:`root`.
        """
    if not node:
        node = self.root
    yield node
    f = self.iterate_all_nodes
    for cnode in node.nodes:
        for ynode in f(cnode):
            yield ynode