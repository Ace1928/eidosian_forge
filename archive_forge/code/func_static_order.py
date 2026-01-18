from types import GenericAlias
def static_order(self):
    """Returns an iterable of nodes in a topological order.

        The particular order that is returned may depend on the specific
        order in which the items were inserted in the graph.

        Using this method does not require to call "prepare" or "done". If any
        cycle is detected, :exc:`CycleError` will be raised.
        """
    self.prepare()
    while self.is_active():
        node_group = self.get_ready()
        yield from node_group
        self.done(*node_group)