from breezy.tests import TestCase
from breezy.textmerge import Merge2
def test_agreed(self):
    lines = 'a\nb\nc\nd\ne\nf\n'.splitlines(True)
    mlines = list(Merge2(lines, lines).merge_lines()[0])
    self.assertEqualDiff(mlines, lines)