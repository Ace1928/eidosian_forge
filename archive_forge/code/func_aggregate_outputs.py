import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def aggregate_outputs(self, runtime=None, needed_outputs=None):
    outputs = self._outputs()
    stdout = runtime.stdout.split('\n')
    outputs.similarity = float(stdout[0])
    return outputs