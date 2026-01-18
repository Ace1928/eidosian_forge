from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def cleanup_headers(self):
    """Make any necessary header changes or defaults

        Subclasses can extend this to add other defaults.
        """
    if 'Content-Length' not in self.headers:
        self.set_content_length()