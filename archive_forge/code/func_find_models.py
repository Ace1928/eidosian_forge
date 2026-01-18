from typing import OrderedDict
import signal
import os
import json
import subprocess
import argparse
import time
import requests
import re
import coolname
def find_models(path, prefix, ep_filter):
    run_name = '_'.join(coolname.generate(2))

    def generate_model_name(root, ep_number):
        return f'{prefix}{os.path.basename(root)}_ep{ep_number}_{run_name}'
    models = {}
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            ep_match = re.match('ep_(\\d+)', d)
            if not ep_match:
                continue
            if ep_filter and ep_match.group(1) != ep_filter:
                continue
            model_name = generate_model_name(root, ep_match.group(1))
            models[model_name] = os.path.join(root, d)
    return OrderedDict(sorted(models.items(), reverse=True, key=lambda x: x[0].split('_ep')[::-1]))