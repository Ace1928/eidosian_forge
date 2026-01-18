import os
import warnings
import re
def readmailcapfile(fp):
    """Read a mailcap file and return a dictionary keyed by MIME type."""
    warnings.warn('readmailcapfile is deprecated, use getcaps instead', DeprecationWarning, 2)
    caps, _ = _readmailcapfile(fp, None)
    return caps