import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
import numpy as np
import lm_eval
from lm_eval import evaluator, tasks
from lm_eval.utils import make_table
def parse_eval_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', '-m', default='hf', help='Name of model, e.g., `hf`.')
    parser.add_argument('--tasks', '-t', default=None, help="Comma-separated list of tasks, or 'list' to display available tasks.")
    parser.add_argument('--model_args', '-a', default='', help='Comma-separated string arguments for model, e.g., `pretrained=EleutherAI/pythia-160m`.')
    parser.add_argument('--open_llm_leaderboard_tasks', '-oplm', action='store_true', default=False, help='Choose the list of tasks with specification in HF open LLM-leaderboard.')
    parser.add_argument('--num_fewshot', '-f', type=int, default=None, help='Number of examples in few-shot context.')
    parser.add_argument('--batch_size', '-b', default=1, help="Batch size, can be 'auto', 'auto:N', or an integer.")
    parser.add_argument('--max_batch_size', type=int, default=None, help="Maximal batch size with 'auto' batch size.")
    parser.add_argument('--device', default=None, help="Device for evaluation, e.g., 'cuda', 'cpu'.")
    parser.add_argument('--output_path', '-o', type=str, default=None, help='Path for saving results.')
    parser.add_argument('--limit', '-L', type=float, default=None, help='Limit number of examples per task.')
    parser.add_argument('--use_cache', '-c', default=None, help='Path to cache db file, if used.')
    parser.add_argument('--verbosity', '-v', default='INFO', help='Logging level: CRITICAL, ERROR, WARNING, INFO, DEBUG.')
    parser.add_argument('--gen_kwargs', default=None, help='Generation kwargs for tasks that support it.')
    parser.add_argument('--check_integrity', action='store_true', help='Whether to run the relevant part of the test suite for the tasks.')
    parser.add_argument('--write_out', '-w', action='store_true', default=False, help='Prints the prompt for the first few documents.')
    parser.add_argument('--log_samples', '-s', action='store_true', default=False, help='If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis.')
    parser.add_argument('--show_config', action='store_true', default=False, help='If True, shows the full config of all tasks at the end of the evaluation.')
    parser.add_argument('--include_path', type=str, default=None, help='Additional path to include if there are external tasks.')
    parser.add_argument('--decontamination_ngrams_path', default=None)
    return parser.parse_args()