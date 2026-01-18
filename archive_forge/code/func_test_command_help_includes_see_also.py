import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_command_help_includes_see_also(self):

    class cmd_WithSeeAlso(commands.Command):
        __doc__ = 'A sample command.'
        _see_also = ['foo', 'bar']
    self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:   brz WithSeeAlso\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nzz{{:See also: bar, foo}}\n', cmd_WithSeeAlso())