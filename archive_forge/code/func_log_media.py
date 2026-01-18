import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
def log_media(self, image: Any, loggable_kwarg_chunks: List, idx: int) -> None:
    """Log the generated images, audio, video, etc. from the Diffusion Pipeline's response along with an optional caption to a media panel in the run.

        Arguments:
            image: (Any) The generated images, audio, video, etc. from the Diffusion
                Pipeline's response.
            loggable_kwarg_chunks: (List) Loggable chunks of kwargs.
        """
    if 'output-type' not in SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]:
        try:
            caption = ''
            if self.pipeline_name in ['StableDiffusionXLPipeline', 'StableDiffusionXLImg2ImgPipeline']:
                prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                prompt2_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt_2')
                caption = f'Prompt-1: {loggable_kwarg_chunks[prompt_index][idx]}\nPrompt-2: {loggable_kwarg_chunks[prompt2_index][idx]}'
            else:
                prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
                caption = loggable_kwarg_chunks[prompt_index][idx]
        except ValueError:
            caption = None
        wandb.log({f'Generated-Image/Pipeline-Call-{self.pipeline_call_count}': wandb.Image(image, caption=caption)})
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'video':
        try:
            prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
            caption = loggable_kwarg_chunks[prompt_index][idx]
        except ValueError:
            caption = None
        wandb.log({f'Generated-Video/Pipeline-Call-{self.pipeline_call_count}': wandb.Video(postprocess_pils_to_np(image), fps=4, caption=caption)})
    elif SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['output-type'] == 'audio':
        try:
            prompt_index = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging'].index('prompt')
            caption = loggable_kwarg_chunks[prompt_index][idx]
        except ValueError:
            caption = None
        wandb.log({f'Generated-Audio/Pipeline-Call-{self.pipeline_call_count}': wandb.Audio(image, sample_rate=16000, caption=caption)})