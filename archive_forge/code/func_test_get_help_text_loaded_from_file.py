import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
def test_get_help_text_loaded_from_file(self):
    topic = help_topics.RegisteredTopic('authentication')
    self.assertStartsWith(topic.get_help_text(), 'Authentication Settings\n=======================\n\n')