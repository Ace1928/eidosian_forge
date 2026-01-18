from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from torch import Tensor
from torchaudio.models import Tacotron2
@abstractmethod
def get_text_processor(self, *, dl_kwargs=None) -> TextProcessor:
    """Create a text processor

        For character-based pipeline, this processor splits the input text by character.
        For phoneme-based pipeline, this processor converts the input text (grapheme) to
        phonemes.

        If a pre-trained weight file is necessary,
        :func:`torch.hub.download_url_to_file` is used to downloaded it.

        Args:
            dl_kwargs (dictionary of keyword arguments,):
                Passed to :func:`torch.hub.download_url_to_file`.

        Returns:
            TextProcessor:
                A callable which takes a string or a list of strings as input and
                returns Tensor of encoded texts and Tensor of valid lengths.
                The object also has ``tokens`` property, which allows to recover the
                tokenized form.

        Example - Character-based
            >>> text = [
            >>>     "Hello World!",
            >>>     "Text-to-speech!",
            >>> ]
            >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
            >>> processor = bundle.get_text_processor()
            >>> input, lengths = processor(text)
            >>>
            >>> print(input)
            tensor([[19, 16, 23, 23, 26, 11, 34, 26, 29, 23, 15,  2,  0,  0,  0],
                    [31, 16, 35, 31,  1, 31, 26,  1, 30, 27, 16, 16, 14, 19,  2]],
                   dtype=torch.int32)
            >>>
            >>> print(lengths)
            tensor([12, 15], dtype=torch.int32)
            >>>
            >>> print([processor.tokens[i] for i in input[0, :lengths[0]]])
            ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']
            >>> print([processor.tokens[i] for i in input[1, :lengths[1]]])
            ['t', 'e', 'x', 't', '-', 't', 'o', '-', 's', 'p', 'e', 'e', 'c', 'h', '!']

        Example - Phoneme-based
            >>> text = [
            >>>     "Hello, T T S !",
            >>>     "Text-to-speech!",
            >>> ]
            >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
            >>> processor = bundle.get_text_processor()
            Downloading:
            100%|███████████████████████████████| 63.6M/63.6M [00:04<00:00, 15.3MB/s]
            >>> input, lengths = processor(text)
            >>>
            >>> print(input)
            tensor([[54, 20, 65, 69, 11, 92, 44, 65, 38,  2,  0,  0,  0,  0],
                    [81, 40, 64, 79, 81,  1, 81, 20,  1, 79, 77, 59, 37,  2]],
                   dtype=torch.int32)
            >>>
            >>> print(lengths)
            tensor([10, 14], dtype=torch.int32)
            >>>
            >>> print([processor.tokens[i] for i in input[0]])
            ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D', '!', '_', '_', '_', '_']
            >>> print([processor.tokens[i] for i in input[1]])
            ['T', 'EH', 'K', 'S', 'T', '-', 'T', 'AH', '-', 'S', 'P', 'IY', 'CH', '!']
        """