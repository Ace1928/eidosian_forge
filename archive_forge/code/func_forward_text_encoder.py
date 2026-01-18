import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def forward_text_encoder(self, texts, dialogue_history=False, batchsize=None):
    """
        Forward pass for a text encoder.

        :param texts:
            text to encode
        :param dialogue_history:
            flag that indicates whether the text is dialogue history; if False,
            text is a response candidate
        :param batchsize:
            size of the batch

        :return:
            encoded representation of the `texts`
        """
    texts_encoded = None
    if texts is None or (dialogue_history and (not self.encode_dialogue_history)):
        if self.multimodal and self.multimodal_combo == 'concat' and dialogue_history:
            texts_encoded = torch.stack([self.blank_encoding for _ in range(batchsize)])
    else:
        encoder = self.context_encoder if dialogue_history else self.label_encoder
        indexes, mask = self.captions_to_tensor(texts)
        texts_encoded = encoder(indexes)
        if self.text_encoder_frozen:
            texts_encoded = texts_encoded.detach()
        texts_encoded = self.additional_layer(texts_encoded)
    return texts_encoded