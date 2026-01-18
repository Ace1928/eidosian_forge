import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def evaluate_one_batch(self, context_encoded, captions_encoded, during_train=False):
    """
        Compute loss - and number of correct examples - for one batch.

        :param context_encoded:
            the encoded context
        :param captions_encoded:
            the encoded captions
        :param during_train:
            true if training, else False

        :return:
            the batch loss and the number of correct examples
        """
    if not during_train:
        self.zero_grad()
        self.eval()
    dot_products = context_encoded.mm(captions_encoded.t())
    log_prob = torch.nn.functional.log_softmax(dot_products, dim=1)
    targets = torch.arange(0, len(context_encoded), dtype=torch.long)
    if self.use_cuda:
        targets = targets.cuda()
    loss = torch.nn.functional.nll_loss(log_prob, targets)
    num_correct = (log_prob.max(dim=1)[1] == targets).float().sum()
    return (loss, num_correct)