import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_topics_for_section(self, section):
    """Get the set of topics in a section."""
    result = set()
    for topic in self.keys():
        if section == self.get_section(topic):
            result.add(topic)
    return result