import enum
import gast
def of(self, node, default=None):
    return getanno(node, self, default=default)