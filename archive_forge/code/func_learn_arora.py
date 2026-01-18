from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch
def learn_arora(opt):
    """
    Go through ConvAI2 data and collect word counts, thus compute the unigram
    probability distribution. Use those probs to compute weighted sentence embeddings
    for all utterances, thus compute first principal component.

    Save all info to arora.pkl file.
    """
    arora_file = os.path.join(opt['datapath'], 'controllable_dialogue', 'arora.pkl')
    opt['task'] = 'fromfile:parlaiformat'
    opt['log_every_n_secs'] = 2
    print('Getting word counts from ConvAI2 train set...')
    opt['datatype'] = 'train:ordered'
    opt['fromfile_datapath'] = os.path.join(opt['datapath'], 'controllable_dialogue', 'ConvAI2_parlaiformat', 'train.txt')
    word_counter_train, total_count_train, all_utts_train = get_word_counts(opt, count_inputs=False)
    print('Getting word counts from ConvAI2 val set...')
    opt['datatype'] = 'valid'
    opt['fromfile_datapath'] = os.path.join(opt['datapath'], 'controllable_dialogue', 'ConvAI2_parlaiformat', 'valid.txt')
    word_counter_valid, total_count_valid, all_utts_valid = get_word_counts(opt, count_inputs=True)
    word_counter = word_counter_train
    for word, count in word_counter_valid.items():
        word_counter[word] += count
    total_count = total_count_train + total_count_valid
    all_utts = all_utts_train + all_utts_valid
    print('Computing unigram probs for all words...')
    word2prob = {w: c / total_count for w, c in word_counter.items()}
    arora_a = 0.0001
    glove_name = '840B'
    glove_dim = 300
    print('Embedding all sentences...')
    sent_embedder = SentenceEmbedder(word2prob, arora_a, glove_name, glove_dim, first_sv=None, data_path=opt['datapath'])
    utt_embs = []
    log_timer = TimeLogger()
    for n, utt in enumerate(all_utts):
        utt_emb = sent_embedder.embed_sent(utt.split(), rem_first_sv=False)
        utt_embs.append(utt_emb)
        if log_timer.time() > opt['log_every_n_secs']:
            text, _log = log_timer.log(n, len(all_utts))
            print(text)
    print('Calculating SVD...')
    utt_embs = np.stack(utt_embs, axis=0)
    U, s, V = np.linalg.svd(utt_embs, full_matrices=False)
    first_sv = V[0, :]
    print('Removing singular vec from all sentence embeddings...')
    utt_embs_adj = [remove_first_sv(torch.Tensor(emb), torch.Tensor(first_sv)).numpy() for emb in utt_embs]
    utt2emb = {utt: emb for utt, emb in zip(all_utts, utt_embs_adj)}
    print('Saving Arora embedding info to %s...' % arora_file)
    with open(arora_file, 'wb') as f:
        pickle.dump({'word2prob': word2prob, 'first_sv': first_sv, 'arora_a': arora_a, 'glove_name': glove_name, 'glove_dim': glove_dim, 'utt2emb': utt2emb}, f)