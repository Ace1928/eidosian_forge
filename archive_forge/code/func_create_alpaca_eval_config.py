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
def create_alpaca_eval_config(alpacaeval_path, model_name):
    config_dir = os.path.join(alpacaeval_path, 'src', 'alpaca_eval', 'models_configs', model_name.lower())
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'configs.yaml')
    config_content = f'{model_name.lower()}:\n  prompt_template: "openchat-13b/prompt.txt"\n  fn_completions: "openai_completions"\n  completions_kwargs:\n    openai_api_base: http://127.0.0.1:18888/v1\n    requires_chatml: True\n    sleep_time: 0\n\n    model_name: "{model_name.lower()}"\n    max_tokens: {MAX_CONTEXT}\n\n    top_p: 1.0\n    temperature: 0.7\n\n    num_procs: 128\n\n  pretty_name: "{model_name}"\n  link: "https://github.com/imoneoi/openchat"\n'
    with open(config_path, 'w') as f:
        f.write(config_content)