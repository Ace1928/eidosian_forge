import os
import sys
import posixpath
import urllib.parse
def read_windows_registry(self, strict=True):
    """
        Load the MIME types database from Windows registry.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
    if not _mimetypes_read_windows_registry and (not _winreg):
        return
    add_type = self.add_type
    if strict:
        add_type = lambda type, ext: self.add_type(type, ext, True)
    if _mimetypes_read_windows_registry:
        _mimetypes_read_windows_registry(add_type)
    elif _winreg:
        self._read_windows_registry(add_type)