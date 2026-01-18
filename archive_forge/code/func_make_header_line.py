from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def make_header_line(is_header, colwidths, colaligns):
    alignment = {'left': '<', 'right': '>', 'center': '^', 'decimal': '>'}
    asciidoc_alignments = zip(colwidths, [alignment[colalign] for colalign in colaligns])
    asciidoc_column_specifiers = ['{:d}{}'.format(width, align) for width, align in asciidoc_alignments]
    header_list = ['cols="' + ','.join(asciidoc_column_specifiers) + '"']
    options_list = []
    if is_header:
        options_list.append('header')
    if options_list:
        header_list += ['options="' + ','.join(options_list) + '"']
    return '[{}]\n|===='.format(','.join(header_list))