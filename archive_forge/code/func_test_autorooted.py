from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_autorooted(self):
    """
        The C{rooted} flag can be updated in some cases, but it cannot be made
        to conflict with other facts surrounding the URL; for example, all URLs
        involving an authority (host) are inherently rooted because it is not
        syntactically possible to express otherwise; also, once an unrooted URL
        gains a path that starts with an empty string, that empty string is
        elided and it becomes rooted, because these cases are syntactically
        indistinguisable in real URL text.
        """
    relative_path_rooted = URL(path=['', 'foo'], rooted=False)
    self.assertEqual(relative_path_rooted.rooted, True)
    relative_flag_rooted = URL(path=['foo'], rooted=True)
    self.assertEqual(relative_flag_rooted.rooted, True)
    self.assertEqual(relative_path_rooted, relative_flag_rooted)
    attempt_unrooted_absolute = URL(host='foo', path=['bar'], rooted=False)
    normal_absolute = URL(host='foo', path=['bar'])
    self.assertEqual(attempt_unrooted_absolute, normal_absolute)
    self.assertEqual(normal_absolute.rooted, True)
    self.assertEqual(attempt_unrooted_absolute.rooted, True)