import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_command_help(self):

    class cmd_Demo(commands.Command):
        __doc__ = 'A sample command.\n\n            :Usage:\n                bzr demo\n\n            :Examples:\n                Example 1::\n\n                    cmd arg1\n\n            Blah Blah Blah\n            '
    export_pot._write_command_help(self.exporter, cmd_Demo())
    result = self.exporter.outf.getvalue()
    result = re.sub('(?m)^#: [^\\n]+\\n', '', result)
    self.assertEqualDiff('msgid "A sample command."\nmsgstr ""\n\nmsgid ""\n":Examples:\\n"\n"    Example 1::"\nmsgstr ""\n\nmsgid "        cmd arg1"\nmsgstr ""\n\nmsgid "Blah Blah Blah"\nmsgstr ""\n\n', result)