import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def add_string(proto, help, maxl, prefix_width=20):
    help_lines = textwrap.wrap(help, maxl - prefix_width, break_long_words=False)
    line_with_indent = '\n' + ' ' * prefix_width
    help_text = line_with_indent.join(help_lines)
    return '%-20s%s\n' % (proto, help_text)