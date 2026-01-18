import sys, os
import textwrap
def format_option_help(self, formatter=None):
    if formatter is None:
        formatter = self.formatter
    formatter.store_option_strings(self)
    result = []
    result.append(formatter.format_heading(_('Options')))
    formatter.indent()
    if self.option_list:
        result.append(OptionContainer.format_option_help(self, formatter))
        result.append('\n')
    for group in self.option_groups:
        result.append(group.format_help(formatter))
        result.append('\n')
    formatter.dedent()
    return ''.join(result[:-1])