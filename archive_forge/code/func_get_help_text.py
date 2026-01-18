import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def get_help_text(self, additional_see_also=None, plain=True):
    """Return a string with the help for this topic.

        :param additional_see_also: Additional help topics to be
            cross-referenced.
        :param plain: if False, raw help (reStructuredText) is
            returned instead of plain text.
        """
    result = topic_registry.get_detail(self.topic)
    result += _format_see_also(additional_see_also)
    if plain:
        result = help_as_plain_text(result)
    i18n.install()
    result = i18n.gettext_per_paragraph(result)
    return result