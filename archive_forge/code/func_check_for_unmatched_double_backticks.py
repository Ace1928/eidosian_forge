from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def check_for_unmatched_double_backticks(self, heading, section):
    """Raise violation if the section contains unmatched double backticks.

    This check counts the number of double backticks in the section and ensures
    that there are an equal number of closing double single-quotes. The common
    mistake is to use a single double-quote to close these values, which breaks
    the rendering. See go/cloud-sdk-help-text#formatting.

    Arguments:
      heading: str, the name of the section.
      section: str, the contents of the section.

    Returns:
      True if there was a violation. None otherwise.
    """
    check_name = self._check_name(heading, 'DOUBLE_BACKTICKS')
    double_backticks_count = len(re.compile('``').findall(section))
    double_single_quotes_count = len(re.compile("''").findall(section))
    unbalanced = double_backticks_count != double_single_quotes_count
    if unbalanced:
        self._add_failure(check_name, 'There are unbalanced double backticks and double single-quotes in the {} section. See go/cloud-sdk-help-text#formatting.'.format(heading))
    else:
        self._add_success(check_name)
    return unbalanced