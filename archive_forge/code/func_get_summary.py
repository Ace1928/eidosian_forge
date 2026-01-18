import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_summary(self, topic):
    """Get the single line summary for the topic."""
    info = self.get_info(topic)
    if info is None:
        return None
    else:
        return info[0]