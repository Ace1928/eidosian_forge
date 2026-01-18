import gettext as _gettext
import os
import sys
def load_plugin_translations(domain):
    """Load the translations for a specific plugin.

    :param domain: Gettext domain name (usually 'brz-PLUGINNAME')
    """
    locale_base = os.path.dirname(__file__)
    translation = install_translations(domain=domain, locale_base=locale_base)
    add_fallback(translation)
    return translation