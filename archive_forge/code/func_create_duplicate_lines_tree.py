import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def create_duplicate_lines_tree(self):
    builder = self.make_branch_builder('branch')
    builder.start_series()
    self.addCleanup(builder.finish_series)
    base_text = b''.join((l for r, l in duplicate_base))
    a_text = b''.join((l for r, l in duplicate_A))
    b_text = b''.join((l for r, l in duplicate_B))
    c_text = b''.join((l for r, l in duplicate_C))
    d_text = b''.join((l for r, l in duplicate_D))
    e_text = b''.join((l for r, l in duplicate_E))
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', base_text))], revision_id=b'rev-base')
    builder.build_snapshot([b'rev-base'], [('modify', ('file', a_text))], revision_id=b'rev-A')
    builder.build_snapshot([b'rev-base'], [('modify', ('file', b_text))], revision_id=b'rev-B')
    builder.build_snapshot([b'rev-A'], [('modify', ('file', c_text))], revision_id=b'rev-C')
    builder.build_snapshot([b'rev-B', b'rev-A'], [('modify', ('file', d_text))], revision_id=b'rev-D')
    builder.build_snapshot([b'rev-C', b'rev-D'], [('modify', ('file', e_text))], revision_id=b'rev-E')
    return builder