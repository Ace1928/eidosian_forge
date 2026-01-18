from typing import Dict, List, Optional
def add_succ(self, id):
    """Add a node id to the node's successors."""
    if isinstance(id, type([])):
        self.succ.extend(id)
    else:
        self.succ.append(id)