import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_help_text_custom_usage(self):
    """Help text may contain a custom usage section."""

    class cmd_Demo(commands.Command):
        __doc__ = 'A sample command.\n\n            :Usage:\n                cmd Demo [opts] args\n\n                cmd Demo -h\n\n            Blah blah blah.\n            '
    self.assertCmdHelp('zz{{:Purpose: zz{{A sample command.}}\n}}zz{{:Usage:\n    zz{{cmd Demo [opts] args}}\n\n    zz{{cmd Demo -h}}\n\n}}\nzz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nDescription:\n  zz{{zz{{Blah blah blah.}}\n\n}}\n', cmd_Demo())