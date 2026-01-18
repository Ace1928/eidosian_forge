from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testHelpWithMember(self):
    with self.assertRaisesFireExit(0, 'SYNOPSIS.*capitalize'):
        core.Fire(tc.TypedProperties, command=['gamma', '--', '--help'])
    with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*capitalize'):
        core.Fire(tc.TypedProperties, command=['gamma', '--help'])
    with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*capitalize'):
        core.Fire(tc.TypedProperties, command=['gamma', '-h'])
    with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*delta'):
        core.Fire(tc.TypedProperties, command=['delta', '--help'])
    with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*echo'):
        core.Fire(tc.TypedProperties, command=['echo', '--help'])