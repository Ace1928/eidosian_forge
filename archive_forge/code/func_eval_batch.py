import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def eval_batch(self, image_features, personalities, captions):
    """
        Evaluate performance of model on one batch.

        Batch is split into chunks of 100 to evaluate hits@1/100

        :param image_features:
            list of tensors of image features
        :param personalities:
            list of personalities
        :param captions:
            list of captions

        :return:
            the total loss, the number of correct examples, and the number of
            examples trained on
        """
    if personalities is None:
        personalities = [''] * len(image_features)
    if len(image_features) == 0:
        return (0, 0, 1)
    self.eval()
    context_encoded, captions_encoded = self.forward(image_features, personalities, captions)
    loss, num_correct, num_examples = self.eval_batch_of_100(context_encoded, captions_encoded)
    return (loss, num_correct, num_examples)