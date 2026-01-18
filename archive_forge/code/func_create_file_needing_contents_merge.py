import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def create_file_needing_contents_merge(self, builder, name):
    file_id = name.encode('ascii') + b'-id'
    transid = builder.add_file(builder.root(), name, b'text1', True, file_id=file_id)
    builder.change_contents(transid, other=b'text4', this=b'text3')