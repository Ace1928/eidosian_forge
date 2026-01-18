from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, \
def get_node_at_pos(self, pos):
    """Get the node at the position (x, y).
        """
    x, y = pos
    for node in self.iterate_open_nodes(self.root):
        if self.x <= x <= self.right and node.y <= y <= node.top:
            return node