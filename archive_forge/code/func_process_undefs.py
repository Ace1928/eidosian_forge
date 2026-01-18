from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
def process_undefs(self):
    for undef_name, _undefs in self._undefs[-1].items():
        if undef_name in self._definitions[-1]:
            for newdef in self._definitions[-1][undef_name]:
                for undef, _ in _undefs:
                    for user in undef.users():
                        newdef.add_user(user)
        else:
            for undef, stars in _undefs:
                if not stars:
                    self.unbound_identifier(undef_name, undef.node)
    self._undefs.pop()