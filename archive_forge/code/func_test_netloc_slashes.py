from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_netloc_slashes(self):
    url = URL.from_text('mailto:mahmoud@hatnote.com')
    self.assertEqual(url.scheme, 'mailto')
    self.assertEqual(url.to_text(), 'mailto:mahmoud@hatnote.com')
    url = URL.from_text('http://hatnote.com')
    self.assertEqual(url.scheme, 'http')
    self.assertEqual(url.to_text(), 'http://hatnote.com')
    url = URL.from_text('newscheme:a:b:c')
    self.assertEqual(url.scheme, 'newscheme')
    self.assertEqual(url.to_text(), 'newscheme:a:b:c')
    url = URL.from_text('newerscheme://a/b/c')
    self.assertEqual(url.scheme, 'newerscheme')
    self.assertEqual(url.to_text(), 'newerscheme://a/b/c')
    url = URL.from_text('git+ftp://gitstub.biz/glyph/lefkowitz')
    self.assertEqual(url.scheme, 'git+ftp')
    self.assertEqual(url.to_text(), 'git+ftp://gitstub.biz/glyph/lefkowitz')
    url = URL.from_text('what+mailto:freerealestate@enotuniq.org')
    self.assertEqual(url.scheme, 'what+mailto')
    self.assertEqual(url.to_text(), 'what+mailto:freerealestate@enotuniq.org')
    url = URL(scheme='ztp', path=('x', 'y', 'z'), rooted=True)
    self.assertEqual(url.to_text(), 'ztp:/x/y/z')
    url = URL(scheme='git+ftp', path=('x', 'y', 'z', ''), rooted=True, uses_netloc=True)
    self.assertEqual(url.to_text(), 'git+ftp:///x/y/z/')
    url = URL.from_text('file:///path/to/heck')
    url2 = url.replace(scheme='mailto')
    self.assertEqual(url2.to_text(), 'mailto:/path/to/heck')
    url_text = 'unregisteredscheme:///a/b/c'
    url = URL.from_text(url_text)
    no_netloc_url = url.replace(uses_netloc=False)
    self.assertEqual(no_netloc_url.to_text(), 'unregisteredscheme:/a/b/c')
    netloc_url = url.replace(uses_netloc=True)
    self.assertEqual(netloc_url.to_text(), url_text)
    return