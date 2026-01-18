from struct import Struct as Packer
from .lib.py3compat import BytesIO, advance_iterator, bchr
from .lib import Container, ListContainer, LazyContainer
def build_stream(self, obj, stream):
    """
        Build an object directly into a stream.
        """
    self._build(obj, stream, Container())