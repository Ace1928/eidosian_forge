import sys
import os
import re
import warnings
import types
import unicodedata
def replace_self(self, new):
    """
        Replace `self` node with `new`, where `new` is a node or a
        list of nodes.
        """
    update = new
    if not isinstance(new, Node):
        try:
            update = new[0]
        except IndexError:
            update = None
    if isinstance(update, Element):
        update.update_basic_atts(self)
    else:
        for att in self.basic_attributes:
            assert not self[att], 'Losing "%s" attribute: %s' % (att, self[att])
    self.parent.replace(self, new)