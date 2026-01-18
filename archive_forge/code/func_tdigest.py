from json import JSONDecoder, JSONEncoder
def tdigest(self):
    """Access the bloom namespace."""
    from .bf import TDigestBloom
    tdigest = TDigestBloom(client=self)
    return tdigest