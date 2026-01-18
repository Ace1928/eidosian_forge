import torch
import torch.nn.functional as F
from typing import Tuple, Any
from parlai.agents.transformer.modules import TransformerGeneratorModel
def decode_forced(self, encoder_states: Tuple[Any, ...], ys: torch.LongTensor) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """
        Decode with a fixed, true sequence, computing loss.

        Overriding `TGM.decode_forced` to bypass assertion that BOS is not present, and
        additionally insert EOS as first token
        """
    bsz = ys.size(0)
    seqlen = ys.size(1)
    inputs = ys.narrow(1, 0, seqlen - 1)
    inputs = torch.cat([torch.LongTensor([self.END_IDX]).detach().expand(bsz, 1).to(inputs), inputs], 1)
    latent, _ = self.decoder(inputs, encoder_states)
    logits = self.output(latent)
    _, preds = logits.max(dim=2)
    return (logits, preds)