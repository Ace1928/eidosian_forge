import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_bugs_topic(topic):
    from breezy import bugtracker
    return 'Bug Tracker Settings\n\n' + bugtracker.tracker_registry.help_topic(topic)