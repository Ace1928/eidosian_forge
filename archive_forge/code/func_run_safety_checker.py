import inspect
import logging
from typing import Callable, List, Optional, Union
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def run_safety_checker(self, image: np.ndarray):
    if self.safety_checker is None:
        has_nsfw_concept = None
    else:
        feature_extractor_input = self.image_processor.numpy_to_pil(image)
        safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors='np').pixel_values.astype(image.dtype)
        images, has_nsfw_concept = ([], [])
        for i in range(image.shape[0]):
            image_i, has_nsfw_concept_i = self.safety_checker(clip_input=safety_checker_input[i:i + 1], images=image[i:i + 1])
            images.append(image_i)
            has_nsfw_concept.append(has_nsfw_concept_i[0])
        image = np.concatenate(images)
    return (image, has_nsfw_concept)