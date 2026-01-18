import warnings
from transformers import AutoTokenizer
from transformers.utils import is_torch_available
from transformers.utils.generic import ExplicitEnum
from ...processing_utils import ProcessorMixin
def char_decode(self, sequences):
    """
        Convert a list of lists of char token ids into a list of strings by calling char tokenizer.

        Args:
            sequences (`torch.Tensor`):
                List of tokenized input ids.
        Returns:
            `List[str]`: The list of char decoded sentences.
        """
    decode_strs = [seq.replace(' ', '') for seq in self.char_tokenizer.batch_decode(sequences)]
    return decode_strs