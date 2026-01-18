import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
def load_inputs_from_json(self, json_file, overwrite=True):
    """
        A convenient way to load pre-set inputs from a JSON file.
        """
    with open(json_file) as fhandle:
        inputs_dict = json.load(fhandle)
    def_inputs = []
    if not overwrite:
        def_inputs = list(self.inputs.get_traitsfree().keys())
    new_inputs = list(set(list(inputs_dict.keys())) - set(def_inputs))
    for key in new_inputs:
        if hasattr(self.inputs, key):
            setattr(self.inputs, key, inputs_dict[key])