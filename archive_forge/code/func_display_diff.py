import hashlib
import os
import re
import sys
from collections import OrderedDict
from optparse import Option, OptionParser
import numpy as np
import nibabel as nib
import nibabel.cmdline.utils
def display_diff(files, diff):
    """Format header differences into a nice string

    Parameters
    ----------
    files: list of files that were compared so we can print their names
    diff: dict of different valued header fields

    Returns
    -------
    str
      string-formatted table of differences
    """
    output = ''
    field_width = '{:<15}'
    filename_width = '{:<53}'
    value_width = '{:<55}'
    output += 'These files are different.\n'
    output += field_width.format('Field/File')
    for i, f in enumerate(files, 1):
        output += '%d:%s' % (i, filename_width.format(os.path.basename(f)))
    output += '\n'
    for key, value in diff.items():
        output += field_width.format(key)
        for item in value:
            if isinstance(item, dict):
                item_str = ', '.join(('%s: %s' % i for i in item.items()))
            elif item is None:
                item_str = '-'
            else:
                item_str = str(item)
            item_str = re.sub('^[ \t]+', '<', item_str)
            item_str = re.sub('[ \t]+$', '>', item_str)
            item_str = re.sub('[\x00]', '?', item_str)
            output += value_width.format(item_str)
        output += '\n'
    return output