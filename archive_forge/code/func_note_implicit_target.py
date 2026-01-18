import sys
import os
import re
import warnings
import types
import unicodedata
def note_implicit_target(self, target, msgnode=None):
    id = self.set_id(target, msgnode)
    self.set_name_id_map(target, id, msgnode, explicit=None)