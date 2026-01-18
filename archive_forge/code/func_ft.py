from json import JSONDecoder, JSONEncoder
def ft(self, index_name='idx'):
    """Access the search namespace, providing support for redis search."""
    from .search import AsyncSearch
    s = AsyncSearch(client=self, index_name=index_name)
    return s