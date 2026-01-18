import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg
def prepare_extra_step_kwargs(self, generator, eta):
    extra_step_kwargs = {}
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    return extra_step_kwargs