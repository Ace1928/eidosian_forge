import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
def splitLine(line_to_split, lines_splitted, maximum_length, split_characters, new_line_characters):
    first_line = True
    while line_to_split and len(line_to_split) > 0:
        split_index = 0
        if len(line_to_split) <= maximum_length:
            split_index = len(line_to_split)
        else:
            for line_index in range(maximum_length):
                if line_to_split[line_index] in split_characters:
                    split_index = line_index + 1
        if split_index == 0:
            split_index = line_index + 1
        if first_line:
            lines_splitted.append(line_to_split[0:split_index])
            first_line = False
            maximum_length -= len(new_line_characters)
        else:
            lines_splitted.append(new_line_characters + line_to_split[0:split_index])
        line_to_split = line_to_split[split_index:]