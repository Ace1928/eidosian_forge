import sys, os
import textwrap
def add_option_group(self, *args, **kwargs):
    if isinstance(args[0], str):
        group = OptionGroup(self, *args, **kwargs)
    elif len(args) == 1 and (not kwargs):
        group = args[0]
        if not isinstance(group, OptionGroup):
            raise TypeError('not an OptionGroup instance: %r' % group)
        if group.parser is not self:
            raise ValueError('invalid OptionGroup (wrong parser)')
    else:
        raise TypeError('invalid arguments')
    self.option_groups.append(group)
    return group