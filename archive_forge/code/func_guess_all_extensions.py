import os
import sys
import posixpath
import urllib.parse
def guess_all_extensions(self, type, strict=True):
    """Guess the extensions for a file based on its MIME type.

        Return value is a list of strings giving the possible filename
        extensions, including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data stream,
        but would be mapped to the MIME type `type' by guess_type().

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
    type = type.lower()
    extensions = list(self.types_map_inv[True].get(type, []))
    if not strict:
        for ext in self.types_map_inv[False].get(type, []):
            if ext not in extensions:
                extensions.append(ext)
    return extensions