from json import JSONDecoder, JSONEncoder
def cf(self):
    """Access the bloom namespace."""
    from .bf import CFBloom
    cf = CFBloom(client=self)
    return cf