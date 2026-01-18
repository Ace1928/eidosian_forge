import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_command_only_additional_see_also(self):

    class cmd_WithSeeAlso(commands.Command):
        __doc__ = 'A sample command.'
    cmd = cmd_WithSeeAlso()
    helptext = cmd.get_help_text(['gam'])
    self.assertEndsWith(helptext, 'zz{{:Options:\n  -h, --help     zz{{Show help message.}}\n  -q, --quiet    zz{{Only display errors and warnings.}}\n  --usage        zz{{Show usage message and options.}}\n  -v, --verbose  zz{{Display more information.}}\n}}\nzz{{:See also: gam}}\n')