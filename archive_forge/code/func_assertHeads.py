from io import StringIO
import testtools
from fastimport import commands, parser
from fastimport.reftracker import RefTracker
from :100
from :101
from :100
from :100
from :100
from :100
from :102
from :100
from :100
from :100
def assertHeads(self, input, expected):
    s = StringIO(input)
    p = parser.ImportParser(s)
    reftracker = RefTracker()
    for cmd in p.iter_commands():
        if isinstance(cmd, commands.CommitCommand):
            reftracker.track_heads(cmd)
            list(cmd.iter_files())
        elif isinstance(cmd, commands.ResetCommand):
            if cmd.from_ is not None:
                reftracker.track_heads_for_ref(cmd.ref, cmd.from_)
    self.assertEqual(reftracker.heads, expected)