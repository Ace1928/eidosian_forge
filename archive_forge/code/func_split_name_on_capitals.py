from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def split_name_on_capitals(collection_name, delimiter=' '):
    """Split camel-cased collection names on capital letters."""
    split_with_spaces = delimiter.join(re.findall('[a-zA-Z][^A-Z]*', collection_name))
    return split_with_spaces