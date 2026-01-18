from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.document_renderers import text_renderer
import six
def check_indentation_for_examples(self):
    if self._prev_heading == 'EXAMPLES' and (not self._buffer.getvalue()):
        self._add_failure(self._check_name('EXAMPLES', 'SECTION_FORMAT'), 'The examples section is not formatted properly. This is likely due to indentation. Please make sure the section is aligned with the heading and not indented.')
        self._example_errors = True