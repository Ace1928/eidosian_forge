from urllib.parse import urlparse
from dulwich.tests import TestCase
from ..config import ConfigDict
from ..credentials import match_partial_url, match_urls, urlmatch_credential_sections
def test_match_partial_url(self):
    url = urlparse('https://github.com/jelmer/dulwich/')
    self.assertTrue(match_partial_url(url, 'github.com'))
    self.assertFalse(match_partial_url(url, 'github.com/jelmer/'))
    self.assertTrue(match_partial_url(url, 'github.com/jelmer/dulwich'))
    self.assertFalse(match_partial_url(url, 'github.com/jel'))
    self.assertFalse(match_partial_url(url, 'github.com/jel/'))