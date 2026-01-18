from .... import branch, errors, osutils, tests
from ....bzr import inventory
from .. import revision_store
from . import FastimportFeature
def content_provider(file_id):
    return b'content of\n' + file_id + b'\n'