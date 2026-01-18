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
def save_inputs_to_json(self, json_file):
    """
        A convenient way to save current inputs to a JSON file.
        """
    inputs = self.inputs.get_traitsfree()
    iflogger.debug('saving inputs %s', inputs)
    with open(json_file, 'w') as fhandle:
        json.dump(inputs, fhandle, indent=4, ensure_ascii=False)