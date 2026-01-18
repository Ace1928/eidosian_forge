import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_no_help_topic(self):
    error = help.NoHelpTopic('topic')
    self.assertEqualDiff("No help could be found for 'topic'. Please use 'brz help topics' to obtain a list of topics.", str(error))