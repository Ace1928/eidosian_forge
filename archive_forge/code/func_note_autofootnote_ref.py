import sys
import os
import re
import warnings
import types
import unicodedata
def note_autofootnote_ref(self, ref):
    self.set_id(ref)
    self.autofootnote_refs.append(ref)