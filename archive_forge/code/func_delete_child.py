import docutils.utils.math.tex2unichar as tex2unichar
def delete_child(self):
    """delete_child() -> child

        Delete last child and return it."""
    child = self.children[-1]
    del self.children[-1]
    return child