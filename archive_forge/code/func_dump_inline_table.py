import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
def dump_inline_table(self, section):
    """Preserve inline table in its compact syntax instead of expanding
        into subsection.

        https://github.com/toml-lang/toml#user-content-inline-table
        """
    retval = ''
    if isinstance(section, dict):
        val_list = []
        for k, v in section.items():
            val = self.dump_inline_table(v)
            val_list.append(k + ' = ' + val)
        retval += '{ ' + ', '.join(val_list) + ' }\n'
        return retval
    else:
        return unicode(self.dump_value(section))