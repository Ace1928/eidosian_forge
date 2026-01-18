import os
from argparse import ArgumentParser, Namespace
from ..data import SingleSentenceClassificationProcessor as Processor
from ..pipelines import TextClassificationPipeline
from ..utils import is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def run_torch(self):
    raise NotImplementedError