from typing import Dict, List, Optional
def remove_succ(self, id):
    """Remove a node id from the node's successors."""
    self.succ.remove(id)