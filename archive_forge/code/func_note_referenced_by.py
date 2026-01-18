import sys
import os
import re
import warnings
import types
import unicodedata
def note_referenced_by(self, name=None, id=None):
    """Note that this Element has been referenced by its name
        `name` or id `id`."""
    self.referenced = 1
    by_name = getattr(self, 'expect_referenced_by_name', {}).get(name)
    by_id = getattr(self, 'expect_referenced_by_id', {}).get(id)
    if by_name:
        assert name is not None
        by_name.referenced = 1
    if by_id:
        assert id is not None
        by_id.referenced = 1