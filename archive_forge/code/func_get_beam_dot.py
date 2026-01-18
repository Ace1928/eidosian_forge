from parlai.core.torch_agent import TorchAgent, Output, Batch
from parlai.utils.misc import round_sigfigs
from parlai.utils.torch import padded_tensor, argsort, neginf
from .modules import Seq2seq, opt_to_kwargs
from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, Counter
from operator import attrgetter
import os
import math
import json
import tempfile
import copy
def get_beam_dot(self, dictionary=None, n_best=None):
    """
        Create pydot graph representation of the beam.

        :param outputs:
            self.outputs from the beam
        :param dictionary:
            tok 2 word dict to save words in the tree nodes
        :returns:
            pydot graph
        """
    try:
        import pydot
    except ImportError:
        print('Please install pydot package to dump beam visualization')
    graph = pydot.Dot(graph_type='digraph')
    outputs = [i.tolist() for i in self.outputs]
    bookkeep = [i.tolist() for i in self.bookkeep]
    all_scores = [i.tolist() for i in self.all_scores]
    if n_best is None:
        n_best = int(self.beam_size / 2)
    top_hyp_idx_n_best = []
    n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
    sorted_finished = self.get_rescored_finished(n_best=n_best)
    for hyptail in sorted_finished:
        top_hyp_idx_n_best.append(self.get_hyp_from_finished(hyptail))
    for tstep, lis in enumerate(outputs):
        for hypid, token in enumerate(lis):
            if tstep == 0:
                hypid = 0
            node_tail = self.HypothesisTail(timestep=tstep, hypid=hypid, score=all_scores[tstep][hypid], tokenid=token)
            color = 'white'
            rank = None
            for i, hypseq in enumerate(top_hyp_idx_n_best):
                if node_tail in hypseq:
                    if n_best <= 5:
                        color = n_best_colors[i]
                    rank = i
                    break
            label = '<{}'.format(dictionary.vec2txt([token]) if dictionary is not None else token) + ' : ' + '{:.{prec}f}>'.format(all_scores[tstep][hypid], prec=3)
            graph.add_node(pydot.Node(node_tail.__repr__(), label=label, fillcolor=color, style='filled', xlabel='{}'.format(rank) if rank is not None else ''))
    for revtstep, lis in reversed(list(enumerate(bookkeep))):
        for i, prev_id in enumerate(lis):
            from_node = graph.get_node('"{}"'.format(self.HypothesisTail(timestep=revtstep, hypid=prev_id, score=all_scores[revtstep][prev_id], tokenid=outputs[revtstep][prev_id]).__repr__()))[0]
            to_node = graph.get_node('"{}"'.format(self.HypothesisTail(timestep=revtstep + 1, hypid=i, score=all_scores[revtstep + 1][i], tokenid=outputs[revtstep + 1][i]).__repr__()))[0]
            newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
            graph.add_edge(newedge)
    return graph