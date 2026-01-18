import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_detail(self, topic):
    """Get the detailed help on a given topic."""
    obj = self.get(topic)
    if callable(obj):
        return obj(topic)
    else:
        return obj