from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def check_for_personal_pronouns(self, heading, section):
    """Raise violation if the section contains personal pronouns."""
    check_name = self._check_name(heading, 'PRONOUN')
    words_in_section = set(re.compile('[\\w/\\-_]+').findall(section.lower()))
    found_pronouns = words_in_section.intersection(self._PERSONAL_PRONOUNS)
    if found_pronouns:
        found_pronouns_list = sorted(list(found_pronouns))
        self._add_failure(check_name, 'Please remove the following personal pronouns in the {} section:\n{}'.format(heading, '\n'.join(found_pronouns_list)))
    else:
        self._add_success(check_name)
    return found_pronouns