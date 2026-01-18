from io import BytesIO
from ... import tests
from .. import pack
def make_parser_expecting_record_type(self):
    parser = pack.ContainerPushParser()
    parser.accept_bytes(b'Bazaar pack format 1 (introduced in 0.18)\n')
    return parser