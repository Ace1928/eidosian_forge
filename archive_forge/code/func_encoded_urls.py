from __future__ import absolute_import
@composite
def encoded_urls(draw):
    """
        A strategy which generates L{EncodedURL}s.
        Call the L{EncodedURL.to_uri} method on each URL to get an HTTP
        protocol-friendly URI.
        """
    port = cast(Optional[int], draw(port_numbers(allow_zero=True)))
    host = cast(Text, draw(hostnames()))
    path = cast(Sequence[Text], draw(paths()))
    if port == 0:
        port = None
    return EncodedURL(scheme=cast(Text, draw(sampled_from((u'http', u'https')))), host=host, port=port, path=path)