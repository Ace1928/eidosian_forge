from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def set_fixed_candidates(self, shared):
    """
        Load a set of fixed candidates and their vectors (or vectorize them here).

        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor

        See the note on the --fixed-candidate-vecs flag for an explanation of the
        'reuse', 'replace', or path options.

        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to additionally perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
    if shared:
        self.fixed_candidates = shared['fixed_candidates']
        self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
        self.fixed_candidate_encs = shared['fixed_candidate_encs']
        self.num_fixed_candidates = shared['num_fixed_candidates']
    else:
        self.num_fixed_candidates = 0
        opt = self.opt
        cand_path = self.fixed_candidates_path
        if 'fixed' in (self.candidates, self.eval_candidates):
            if not cand_path:
                path = self.get_task_candidates_path()
                if path:
                    logging.info(f'setting fixed_candidates path to: {path}')
                    self.fixed_candidates_path = path
                    cand_path = self.fixed_candidates_path
            logging.info(f'Loading fixed candidate set from {cand_path}')
            with open(cand_path, 'r', encoding='utf-8') as f:
                cands = [line.strip() for line in f.readlines()]
            if os.path.isfile(self.opt['fixed_candidate_vecs']):
                vecs_path = opt['fixed_candidate_vecs']
                vecs = self.load_candidates(vecs_path)
            else:
                setting = self.opt['fixed_candidate_vecs']
                model_dir, model_file = os.path.split(self.opt['model_file'])
                model_name = os.path.splitext(model_file)[0]
                cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                vecs_path = os.path.join(model_dir, '.'.join([model_name, cands_name, 'vecs']))
                if setting == 'reuse' and os.path.isfile(vecs_path):
                    vecs = self.load_candidates(vecs_path)
                else:
                    vecs = self._make_candidate_vecs(cands)
                    self._save_candidates(vecs, vecs_path)
            self.fixed_candidates = cands
            self.num_fixed_candidates = len(self.fixed_candidates)
            self.fixed_candidate_vecs = vecs
            if self.use_cuda:
                self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()
            if self.encode_candidate_vecs:
                enc_path = os.path.join(model_dir, '.'.join([model_name, cands_name, 'encs']))
                if setting == 'reuse' and os.path.isfile(enc_path):
                    encs = self.load_candidates(enc_path, cand_type='encodings')
                else:
                    encs = self._make_candidate_encs(self.fixed_candidate_vecs)
                    self._save_candidates(encs, path=enc_path, cand_type='encodings')
                self.fixed_candidate_encs = encs
                if self.use_cuda:
                    self.fixed_candidate_encs = self.fixed_candidate_encs.cuda()
                if self.fp16:
                    self.fixed_candidate_encs = self.fixed_candidate_encs.half()
                else:
                    self.fixed_candidate_encs = self.fixed_candidate_encs.float()
            else:
                self.fixed_candidate_encs = None
        else:
            self.fixed_candidates = None
            self.fixed_candidate_vecs = None
            self.fixed_candidate_encs = None