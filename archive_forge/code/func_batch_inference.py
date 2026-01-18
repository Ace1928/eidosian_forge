import collections
import types
import numpy as np
from ..utils import (
from .base import ArgumentHandler, Dataset, Pipeline, PipelineException, build_pipeline_init_args
def batch_inference(self, **inputs):
    return self.model(**inputs)