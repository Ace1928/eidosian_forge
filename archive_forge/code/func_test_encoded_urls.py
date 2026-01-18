@given(encoded_urls())
def test_encoded_urls(self, url):
    """
            encoded_urls() generates EncodedURLs.
            """
    self.assertIsInstance(url, EncodedURL)