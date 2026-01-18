import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_get_topics_topics(self):
    """Searching for a string returns the matching string."""
    index = help_topics.HelpTopicIndex()
    topics = index.get_topics('topics')
    self.assertEqual(1, len(topics))
    self.assertIsInstance(topics[0], help_topics.RegisteredTopic)
    self.assertEqual('topics', topics[0].topic)