import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_current_formats_topic(topic):
    from breezy import controldir
    return 'Current Storage Formats\n\n' + controldir.format_registry.help_topic(topic)