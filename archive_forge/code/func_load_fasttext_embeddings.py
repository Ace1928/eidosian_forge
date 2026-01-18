import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def load_fasttext_embeddings(dic, embedding_dim, datapath):
    """
    Load weights from fasttext_cc and put them in embeddings.weights.
    """
    print('Initializing embeddings from fasttext_cc')
    from parlai.zoo.fasttext_cc_vectors.build import download
    pretrained = download(datapath)
    print('Done Loading vectors from fasttext. {} embeddings loaded.'.format(len(pretrained)))
    used = 0
    res = nn.Embedding(len(dic), embedding_dim)
    for word in dic.tok2ind.keys():
        index = dic.tok2ind[word]
        if word in pretrained and res.weight.data.shape[0] > index:
            res.weight.data[index] = pretrained[word]
            used += 1
    print('{} have been initialized on pretrained over {} words'.format(used, len(dic)))
    return res