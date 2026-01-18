import urlparse
def test_iri(self):
    """Test that the right type of escaping is done for each part of the URI."""
    self.assertEqual('http://xn--o3h.com/%E2%98%84', iri2uri(u'http://☄.com/☄'))
    self.assertEqual('http://bitworking.org/?fred=%E2%98%84', iri2uri(u'http://bitworking.org/?fred=☄'))
    self.assertEqual('http://bitworking.org/#%E2%98%84', iri2uri(u'http://bitworking.org/#☄'))
    self.assertEqual('#%E2%98%84', iri2uri(u'#☄'))
    self.assertEqual('/fred?bar=%E2%98%9A#%E2%98%84', iri2uri(u'/fred?bar=☚#☄'))
    self.assertEqual('/fred?bar=%E2%98%9A#%E2%98%84', iri2uri(iri2uri(u'/fred?bar=☚#☄')))
    self.assertNotEqual('/fred?bar=%E2%98%9A#%E2%98%84', iri2uri(u'/fred?bar=☚#☄'.encode('utf-8')))