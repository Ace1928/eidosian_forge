import re
from io import BytesIO
from .. import errors
def iter_records_from_file(source_file):
    parser = ContainerPushParser()
    while True:
        bytes = source_file.read(parser.read_size_hint())
        parser.accept_bytes(bytes)
        yield from parser.read_pending_records()
        if parser.finished:
            break