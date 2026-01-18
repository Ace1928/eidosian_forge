import re
from json import scanner
def raw_decode(self, s, idx=0):
    """Decode a JSON document from ``s`` (a ``str`` beginning with
        a JSON document) and return a 2-tuple of the Python
        representation and the index in ``s`` where the document ended.

        This can be used to decode a JSON document from a string that may
        have extraneous data at the end.

        """
    try:
        obj, end = self.scan_once(s, idx)
    except StopIteration as err:
        raise JSONDecodeError('Expecting value', s, err.value) from None
    return (obj, end)