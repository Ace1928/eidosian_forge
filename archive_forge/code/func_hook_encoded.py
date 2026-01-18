import io
import sys, os
from types import GenericAlias
def hook_encoded(encoding, errors=None):

    def openhook(filename, mode):
        return open(filename, mode, encoding=encoding, errors=errors)
    return openhook