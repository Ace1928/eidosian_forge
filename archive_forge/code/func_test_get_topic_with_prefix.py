import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_get_topic_with_prefix(self):
    """Searching for commands/rocks returns the rocks command object."""
    index = commands.HelpCommandIndex()
    topics = index.get_topics('commands/rocks')
    self.assertEqual(1, len(topics))
    self.assertIsInstance(topics[0], builtins.cmd_rocks)