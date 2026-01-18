import os
import sys
import posixpath
import urllib.parse
def guess_extension(self, type, strict=True):
    """Guess the extension for a file based on its MIME type.

        Return value is a string giving a filename extension,
        including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data
        stream, but would be mapped to the MIME type `type' by
        guess_type().  If no extension can be guessed for `type', None
        is returned.

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
    extensions = self.guess_all_extensions(type, strict)
    if not extensions:
        return None
    return extensions[0]