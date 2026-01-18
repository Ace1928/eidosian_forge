import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def initialize_control_information(opt, build_task=True):
    """
    Loads information from word2count.pkl, arora.pkl in data/controllable_dialogue, and
    uses it to initialize objects for computing NIDF and response-relatedness controls.

    By default (build_task=True) we will also build the controllable_dialogue task i.e.
    download data/controllable_dialogue if necessary.
    """
    global word2nidf, nidf_feats, arora_data, sent_embedder
    if word2nidf is not None:
        return
    if build_task:
        build(opt)
    print('Loading up controllable features...')
    word2nidf = load_word2nidf(opt)
    nidf_feats = NIDFFeats()
    arora_data = load_arora(opt)
    sent_embedder = SentenceEmbedder(arora_data['word2prob'], arora_data['arora_a'], arora_data['glove_name'], arora_data['glove_dim'], arora_data['first_sv'], data_path=opt['datapath'])