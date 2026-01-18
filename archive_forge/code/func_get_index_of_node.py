from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def get_index_of_node(self, node, selectable_nodes):
    """(internal) Returns the index of the `node` within the
        `selectable_nodes` returned by :meth:`get_selectable_nodes`.
        """
    return selectable_nodes.index(node)