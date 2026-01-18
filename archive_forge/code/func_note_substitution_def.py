import sys
import os
import re
import warnings
import types
import unicodedata
def note_substitution_def(self, subdef, def_name, msgnode=None):
    name = whitespace_normalize_name(def_name)
    if name in self.substitution_defs:
        msg = self.reporter.error('Duplicate substitution definition name: "%s".' % name, base_node=subdef)
        if msgnode != None:
            msgnode += msg
        oldnode = self.substitution_defs[name]
        dupname(oldnode, name)
    self.substitution_defs[name] = subdef
    self.substitution_names[fully_normalize_name(name)] = name