import os, sys, types, json, math, time
import numpy as np
import torch
from torch.nn import functional as F
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from lm_eval import tasks, evaluator
from lm_eval.models.gpt2 import GPT2LM
@torch.no_grad()
def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
    results = evaluator.evaluate(lm=self, task_dict=tasks.get_task_dict(eval_tasks), provide_description=False, num_fewshot=num_fewshot, limit=None, bootstrap_iters=bootstrap_iters)
    return results