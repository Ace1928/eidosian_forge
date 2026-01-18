import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def annotateEqual(self, expected, parents, newlines, revision_id, blocks=None):
    annotate_list = list(annotate.reannotate(parents, newlines, revision_id, blocks))
    self.assertEqual(len(expected), len(annotate_list))
    for e, a in zip(expected, annotate_list):
        self.assertEqual(e, a)