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
def run_alpaca_eval(alpacaeval_path, model_name):
    if os.path.exists(os.path.join(alpacaeval_path, 'results', model_name.lower())):
        return
    create_alpaca_eval_config(alpacaeval_path, model_name)
    command = f'python -m alpaca_eval.main evaluate_from_model --model_configs {model_name.lower()} --annotators_config alpaca_eval_gpt4'
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{env.get('PYTHONPATH', '')}:{os.path.join(alpacaeval_path, 'src')}'
    subprocess.run(command, shell=True, cwd=alpacaeval_path, env=env)