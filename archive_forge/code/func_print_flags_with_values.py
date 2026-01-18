from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.flags import _exceptions
def print_flags_with_values(self, flag_values):
    prefix = 'flags '
    flags_with_values = []
    for key in self.flag_names:
        flags_with_values.append('%s=%s' % (key, flag_values[key].value))
    return prefix + ', '.join(flags_with_values)