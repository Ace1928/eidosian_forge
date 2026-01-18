import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
def get_speech_features(self, speech_ids: Optional[torch.LongTensor]=None, input_ids: Optional[torch.LongTensor]=None, input_features: Optional[torch.FloatTensor]=None, conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, **kwargs) -> torch.FloatTensor:
    """
        This method can be used to extract speech_embeds. The speech embeddings are obtained by applying the speech
        model on speech_ids. If speech_ids is not present but both input_ids and input_features are given then the
        decoder model will be used to first generate the speech_ids and then applying the speech model.

        Args:
            speech_ids (`torch.LongTensor` of shape `(batch_size, num_speech_ids)`, *optional*):
                Speech Tokens. Padding will be ignored by default should you provide it. If speech_ids are provided
                then input_ids and input_features will be automatically ignored.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Input text Tokens. Processed from the [`ClvpTokenizer`]. If speech_ids is not provided, then input_ids
                and input_features will be used.
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`, *optional*):
                Indicates log-melspectrogram representations for audio returned by [`ClvpFeatureExtractor`]. If
                speech_ids is not provided, then input_ids and input_features will be used.
            conditioning_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
                inputs_embeds for `ClvpConditioningEncoder`. Can be used in place of `input_ids`.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding speech token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            generation_config (`GenerationConfig`, *optional*):
                generation config to control the generation of speech_ids if they are not provided.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The speech embeddings obtained by applying the projection layer to the pooled output of the CLVP Speech
                Model.

        Examples:

        ```python
        >>> import datasets
        >>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

        >>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
        >>> text = "This is an example text."
        >>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        >>> _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        >>> # Define processor and model
        >>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and model output
        >>> processor_output = processor(raw_speech=audio, sampling_rate=sr, text=text, return_tensors="pt")
        >>> speech_embeds = model.get_speech_features(
        ...     input_ids=processor_output["input_ids"], input_features=processor_output["input_features"]
        ... )
        ```
        """
    if speech_ids is None:
        if input_ids is None and conditioning_encoder_inputs_embeds is None or input_features is None:
            raise ValueError('Either speech_ids or input_ids/conditioning_encoder_inputs_embeds and input_features must be provided.')
        if generation_config is None:
            generation_config = self.generation_config
        generation_config.update(**kwargs)
        conditioning_embeds = self.conditioning_encoder(input_features=input_features, input_ids=input_ids, inputs_embeds=conditioning_encoder_inputs_embeds, attention_mask=attention_mask)
        speech_ids = self.speech_decoder_model.generate(conditioning_embeds=conditioning_embeds, generation_config=generation_config)
        speech_ids = self.fix_speech_decoder_output(speech_ids[0])
    outputs = self.speech_encoder_model(input_ids=speech_ids, attention_mask=attention_mask)
    return outputs[0]