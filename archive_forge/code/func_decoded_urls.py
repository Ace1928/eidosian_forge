from __future__ import absolute_import
@composite
def decoded_urls(draw):
    """
        A strategy which generates L{DecodedURL}s.
        Call the L{EncodedURL.to_uri} method on each URL to get an HTTP
        protocol-friendly URI.
        """
    return DecodedURL(draw(encoded_urls()))