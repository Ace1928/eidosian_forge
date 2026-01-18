import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
def serialize_text(self):
    """Returns a serialized form of the Namepace.

        All the elements in the namespace are sorted by
        URI, joined to the associated prefix with a colon and
        separated with spaces.
        :return: bytes
        """
    if self._uri_to_prefix is None or len(self._uri_to_prefix) == 0:
        return b''
    od = collections.OrderedDict(sorted(self._uri_to_prefix.items()))
    data = []
    for uri in od:
        data.append(uri + ':' + od[uri])
    return ' '.join(data).encode('utf-8')