import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def demonstrate_ansi_formatting():
    """Demonstrate the use of ANSI escape sequences."""
    output('%s', ansi_wrap('Text styles:', bold=True))
    styles = ['normal', 'bright']
    styles.extend(ANSI_TEXT_STYLES.keys())
    for style_name in sorted(styles):
        options = dict(color=HIGHLIGHT_COLOR)
        if style_name != 'normal':
            options[style_name] = True
        style_label = style_name.replace('_', ' ').capitalize()
        output(' - %s', ansi_wrap(style_label, **options))
    for color_type, color_label in (('color', 'Foreground colors'), ('background', 'Background colors')):
        intensities = [('normal', dict()), ('bright', dict(bright=True))]
        if color_type != 'background':
            intensities.insert(0, ('faint', dict(faint=True)))
        output('\n%s' % ansi_wrap('%s:' % color_label, bold=True))
        output(format_smart_table([[color_name] + [ansi_wrap('XXXXXX' if color_type != 'background' else ' ' * 6, **dict(list(kw.items()) + [(color_type, color_name)])) for label, kw in intensities] for color_name in sorted(ANSI_COLOR_CODES.keys())], column_names=['Color'] + [label.capitalize() for label, kw in intensities]))
    demonstrate_256_colors(0, 7, 'standard colors')
    demonstrate_256_colors(8, 15, 'high-intensity colors')
    demonstrate_256_colors(16, 231, '216 colors')
    demonstrate_256_colors(232, 255, 'gray scale colors')