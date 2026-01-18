import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_parlai_args(self, args=None):
    """
        Add common ParlAI args across all scripts.
        """
    self.add_argument('--helpall', action='helpall', help='Show usage, including advanced arguments.')
    parlai = self.add_argument_group('Main ParlAI Arguments')
    parlai.add_argument('-o', '--init-opt', default=None, help='Path to json file of options. Note: Further Command-line arguments override file-based options.')
    parlai.add_argument('-t', '--task', help='ParlAI task(s), e.g. "babi:Task1" or "babi,cbt"')
    parlai.add_argument('--download-path', default=None, hidden=True, help='path for non-data dependencies to store any needed files.defaults to {parlai_dir}/downloads')
    parlai.add_argument('--loglevel', default='info', hidden=True, choices=logging.get_all_levels(), help='Logging level')
    parlai.add_argument('-dt', '--datatype', metavar='DATATYPE', default='train', choices=['train', 'train:stream', 'train:ordered', 'train:ordered:stream', 'train:stream:ordered', 'train:evalmode', 'train:evalmode:stream', 'train:evalmode:ordered', 'train:evalmode:ordered:stream', 'train:evalmode:stream:ordered', 'valid', 'valid:stream', 'test', 'test:stream'], help='choose from: train, train:ordered, valid, test. to stream data add ":stream" to any option (e.g., train:stream). by default train is random with replacement, valid is ordered, test is ordered.')
    parlai.add_argument('-im', '--image-mode', default='raw', type=str, help='image preprocessor to use. default is "raw". set to "none" to skip image loading.', hidden=True)
    parlai.add_argument('--hide-labels', default=False, type='bool', hidden=True, help='default (False) moves labels in valid and test sets to the eval_labels field. If True, they are hidden completely.')
    parlai.add_argument('-mtw', '--multitask-weights', type='multitask_weights', default=[1], help='list of floats, one for each task, specifying the probability of drawing the task in multitask case. You may also provide "stochastic" to simulate simple concatenation.', hidden=True)
    parlai.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size for minibatch training schemes')
    parlai.add_argument('-dynb', '--dynamic-batching', default=None, type='nonestr', choices={None, 'full', 'batchsort'}, help='Use dynamic batching')
    self.add_parlai_data_path(parlai)