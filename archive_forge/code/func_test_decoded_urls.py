@given(decoded_urls())
def test_decoded_urls(self, url):
    """
            decoded_urls() generates DecodedURLs.
            """
    self.assertIsInstance(url, DecodedURL)