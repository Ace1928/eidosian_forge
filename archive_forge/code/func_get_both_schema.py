import parlai.core.build_data as build_data
import os
import json
import re
from parlai.core.build_data import DownloadableFile
def get_both_schema(context):
    variations = [x[1:-1].split('/') for x in re.findall(pattern, context)]
    splits = re.split(pattern, context)
    results = []
    for which_schema in range(2):
        vs = [v[which_schema] for v in variations]
        context = ''
        for idx in range(len(splits)):
            context += splits[idx]
            if idx < len(vs):
                context += vs[idx]
        results.append(context)
    return results