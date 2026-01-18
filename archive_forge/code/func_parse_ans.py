import parlai.core.build_data as build_data
import json
import os
import re
from parlai.core.build_data import DownloadableFile
def parse_ans(a):
    a = a.lstrip('(list')
    ans = ''
    for a in re.split('\\(description', a):
        a = a.strip(STRIP_CHARS)
        ans = ans + '|' + a
    return ans.lstrip('|')