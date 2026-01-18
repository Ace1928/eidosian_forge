import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def assertCmdHelp(self, expected, cmd):
    self.assertEqualDiff(textwrap.dedent(expected), cmd.get_help_text())