from .. import tests
from . import features
def assertChunksToLines(self, lines, chunks, alreadly_lines=False):
    result = self.module.chunks_to_lines(chunks)
    self.assertEqual(lines, result)
    if alreadly_lines:
        self.assertIs(chunks, result)