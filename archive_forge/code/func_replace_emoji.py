import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
def replace_emoji(x):
    if x in UNICODE_EMOJI.keys():
        return ' ' + UNICODE_EMOJI[x].replace(':', '@') + ' '
    else:
        return x