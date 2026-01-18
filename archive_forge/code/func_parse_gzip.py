import parlai.core.build_data as build_data
import os
import gzip
import json
from parlai.core.build_data import DownloadableFile
def parse_gzip(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))