import sys
import os
import re
import warnings
import types
import unicodedata
def note_footnote_ref(self, ref):
    self.set_id(ref)
    self.footnote_refs.setdefault(ref['refname'], []).append(ref)
    self.note_refname(ref)