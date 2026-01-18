import os
from breezy.tests import TestCaseWithTransport
from breezy.version_info_formats import VersionInfoBuilder
def assertEqualNoBuildDate(self, text1, text2):
    """Compare 2 texts, but ignore the build-date field.

        build-date is the current timestamp, accurate to seconds. But the
        clock is always ticking, and it may have ticked between the time
        that text1 and text2 were generated.
        """
    lines1 = text1.splitlines(True)
    lines2 = text2.splitlines(True)
    for line1, line2 in zip(lines1, lines2):
        if line1.startswith('build-date: '):
            self.assertStartsWith(line2, 'build-date: ')
        else:
            self.assertEqual(line1, line2)
    self.assertEqual(len(lines1), len(lines2))