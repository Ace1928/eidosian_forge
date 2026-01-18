import sys
import logging; log = logging.getLogger(__name__)
from types import ModuleType
def join_byte_values(values):
    return join_bytes((chr(v) for v in values))