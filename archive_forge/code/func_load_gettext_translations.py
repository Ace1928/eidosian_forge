import codecs
import csv
import datetime
import gettext
import glob
import os
import re
from tornado import escape
from tornado.log import gen_log
from tornado._locale_data import LOCALE_NAMES
from typing import Iterable, Any, Union, Dict, Optional
def load_gettext_translations(directory: str, domain: str) -> None:
    """Loads translations from `gettext`'s locale tree

    Locale tree is similar to system's ``/usr/share/locale``, like::

        {directory}/{lang}/LC_MESSAGES/{domain}.mo

    Three steps are required to have your app translated:

    1. Generate POT translation file::

        xgettext --language=Python --keyword=_:1,2 -d mydomain file1.py file2.html etc

    2. Merge against existing POT file::

        msgmerge old.po mydomain.po > new.po

    3. Compile::

        msgfmt mydomain.po -o {directory}/pt_BR/LC_MESSAGES/mydomain.mo
    """
    global _translations
    global _supported_locales
    global _use_gettext
    _translations = {}
    for filename in glob.glob(os.path.join(directory, '*', 'LC_MESSAGES', domain + '.mo')):
        lang = os.path.basename(os.path.dirname(os.path.dirname(filename)))
        try:
            _translations[lang] = gettext.translation(domain, directory, languages=[lang])
        except Exception as e:
            gen_log.error("Cannot load translation for '%s': %s", lang, str(e))
            continue
    _supported_locales = frozenset(list(_translations.keys()) + [_default_locale])
    _use_gettext = True
    gen_log.debug('Supported locales: %s', sorted(_supported_locales))