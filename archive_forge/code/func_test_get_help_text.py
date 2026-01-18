import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_get_help_text(self):
    """RegisteredTopic returns the get_detail results for get_help_text."""
    topic = help_topics.RegisteredTopic('commands')
    self.assertEqual(help_topics.topic_registry.get_detail('commands'), topic.get_help_text())