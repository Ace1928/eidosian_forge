from json import JSONDecoder, JSONEncoder
def cms(self):
    """Access the bloom namespace."""
    from .bf import CMSBloom
    cms = CMSBloom(client=self)
    return cms