import logging
from typing import Any, Dict, List, Sequence
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from .utils import (
def prepare_loggable_dict(self, pipeline: Any, response: Response, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the loggable dictionary, which is the packed data as a dictionary for logging to wandb, None if an exception occurred.

        Arguments:
            pipeline: (Any) The Diffusion Pipeline.
            response: (wandb.sdk.integration_utils.auto_logging.Response) The response from
                the request.
            kwargs: (Dict[str, Any]) Dictionary of keyword arguments.

        Returns:
            Packed data as a dictionary for logging to wandb, None if an exception occurred.
        """
    images = self.get_output_images(response)
    if self.pipeline_name == 'StableDiffusionXLPipeline' and kwargs['output_type'] == 'latent':
        images = decode_sdxl_t2i_latents(pipeline, response.images)
    if self.pipeline_name in ['TextToVideoSDPipeline', 'TextToVideoZeroPipeline']:
        video = postprocess_np_arrays_for_video(images, normalize=self.pipeline_name == 'TextToVideoZeroPipeline')
        wandb.log({f'Generated-Video/Pipeline-Call-{self.pipeline_call_count}': wandb.Video(video, fps=4, caption=kwargs['prompt'])})
        loggable_kwarg_ids = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging']
        table_row = [kwargs[loggable_kwarg_ids[idx]] for idx in range(len(loggable_kwarg_ids))]
        table_row.append(wandb.Video(video, fps=4))
        self.wandb_table.add_data(*table_row)
    else:
        loggable_kwarg_ids = SUPPORTED_MULTIMODAL_PIPELINES[self.pipeline_name]['kwarg-logging']
        loggable_kwarg_chunks = []
        for loggable_kwarg_id in loggable_kwarg_ids:
            loggable_kwarg_chunks.append(kwargs[loggable_kwarg_id] if isinstance(kwargs[loggable_kwarg_id], list) else [kwargs[loggable_kwarg_id]])
        images = chunkify(images, len(loggable_kwarg_chunks[0]))
        for idx in range(len(loggable_kwarg_chunks[0])):
            for image in images[idx]:
                self.log_media(image, loggable_kwarg_chunks, idx)
                self.add_data_to_table(image, loggable_kwarg_chunks, idx)
    return {f'Result-Table/Pipeline-Call-{self.pipeline_call_count}': self.wandb_table}