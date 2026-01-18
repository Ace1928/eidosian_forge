import os
import sys
import posixpath
import urllib.parse
def add_type(self, type, ext, strict=True):
    """Add a mapping between a type and an extension.

        When the extension is already known, the new
        type will replace the old one. When the type
        is already known the extension will be added
        to the list of known extensions.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
    self.types_map[strict][ext] = type
    exts = self.types_map_inv[strict].setdefault(type, [])
    if ext not in exts:
        exts.append(ext)