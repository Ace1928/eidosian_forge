from typing import Dict, List, Optional
def is_parent_of(self, parent, grandchild):
    """Check if grandchild is a subnode of parent."""
    if grandchild == parent or grandchild in self.chain[parent].get_succ():
        return True
    else:
        for sn in self.chain[parent].get_succ():
            if self.is_parent_of(sn, grandchild):
                return True
        else:
            return False