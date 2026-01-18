from .. import annotate, errors, revision, tests
from ..bzr import knit
def custom_tiebreaker(annotated_lines):
    self.assertEqual(2, len(annotated_lines))
    left = annotated_lines[0]
    self.assertEqual(2, len(left))
    self.assertEqual(b'new content\n', left[1])
    right = annotated_lines[1]
    self.assertEqual(2, len(right))
    self.assertEqual(b'new content\n', right[1])
    seen.update([left[0], right[0]])
    if left[0] < right[0]:
        return right
    else:
        return left