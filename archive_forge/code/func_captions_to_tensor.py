import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def captions_to_tensor(self, captions):
    """
        Tokenize a list of sentences into a 2D float tensor.

        :param captions:
            list of sentences to tokenize

        :return:
            a (batchsize X truncate_length) tensor representation of the captions,
            and a similarly sized mask tensor
        """
    max_length = self.truncate_length
    indexes = []
    for c in captions:
        vec = self.dictionary.txt2vec(c)
        if len(vec) > max_length:
            vec = vec[:max_length]
        indexes.append(self.dictionary.txt2vec(c))
    longest = max([len(v) for v in indexes])
    res = torch.LongTensor(len(captions), longest).fill_(self.dictionary.tok2ind[self.dictionary.null_token])
    mask = torch.FloatTensor(len(captions), longest).fill_(0)
    for i, inds in enumerate(indexes):
        res[i, 0:len(inds)] = torch.LongTensor(inds)
        mask[i, 0:len(inds)] = torch.FloatTensor([1] * len(inds))
    if self.use_cuda:
        res = res.cuda()
        mask = mask.cuda()
    return (res, mask)