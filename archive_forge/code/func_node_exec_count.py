import os
@property
def node_exec_count(self):
    return sum(self._node_to_exec_count.values())