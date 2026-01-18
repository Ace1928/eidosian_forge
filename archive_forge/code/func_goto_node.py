from time import time
from os import environ
from kivy.properties import NumericProperty, BooleanProperty, ListProperty
def goto_node(self, key, last_node, last_node_idx):
    """(internal) Used by the controller to get the node at the position
        indicated by key. The key can be keyboard inputs, e.g. pageup,
        or scroll inputs from the mouse scroll wheel, e.g. scrollup.
        'last_node' is the last node selected and is used to find the resulting
        node. For example, if the key is up, the returned node is one node
        up from the last node.

        It can be overwritten by the derived widget.

        :Parameters:
            `key`
                str, the string used to find the desired node. It can be any
                of the keyboard keys, as well as the mouse scrollup,
                scrolldown, scrollright, and scrollleft strings. If letters
                are typed in quick succession, the letters will be combined
                before it's passed in as key and can be used to find nodes that
                have an associated string that starts with those letters.
            `last_node`
                The last node that was selected.
            `last_node_idx`
                The cached index of the last node selected in the
                :meth:`get_selectable_nodes` list. If the list hasn't changed
                it saves having to look up the index of `last_node` in that
                list.

        :Returns:
            tuple, the node targeted by key and its index in the
            :meth:`get_selectable_nodes` list. Returning
            `(last_node, last_node_idx)` indicates a node wasn't found.
        """
    sister_nodes = self.get_selectable_nodes()
    end = len(sister_nodes) - 1
    counts = self._offset_counts
    if end == -1:
        return (last_node, last_node_idx)
    if last_node_idx > end or sister_nodes[last_node_idx] != last_node:
        try:
            last_node_idx = self.get_index_of_node(last_node, sister_nodes)
        except ValueError:
            return (last_node, last_node_idx)
    is_reversed = self.nodes_order_reversed
    if key in counts:
        count = -counts[key] if is_reversed else counts[key]
        idx = max(min(count + last_node_idx, end), 0)
        return (sister_nodes[idx], idx)
    elif key == 'home':
        if is_reversed:
            return (sister_nodes[end], end)
        return (sister_nodes[0], 0)
    elif key == 'end':
        if is_reversed:
            return (sister_nodes[0], 0)
        return (sister_nodes[end], end)
    else:
        return (last_node, last_node_idx)