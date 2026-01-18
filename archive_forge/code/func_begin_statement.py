import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def begin_statement(self, stmt):
    """Marks the beginning of a statement.

    Args:
      stmt: Hashable, a key by which the statement can be identified in the
        CFG's stmt_prev and stmt_next attributes
    """
    self.active_stmts.add(stmt)