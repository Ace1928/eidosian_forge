from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def deselect_node(self, node):
    """ Deselects a possibly selected node.

        It is called by the controller when it deselects a node and can also
        be called from the outside to deselect a node directly. The derived
        widget should overwrite this method and change the node to its
        unselected state when this is called

        :Parameters:
            `node`
                The node to be deselected.

        .. warning::

            This method must be called by the derived widget using super if it
            is overwritten.
        """
    try:
        self.selected_nodes.remove(node)
        return True
    except ValueError:
        return False