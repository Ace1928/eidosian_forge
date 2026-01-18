import inspect
from typing import Any, Dict, List, Optional, Union
import torch.nn
from . import utils, variables
from .bytecode_transformation import (
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
def codegen_save_tempvars(self, cg: PyCodegen):
    for var in self._get_modified_vars():
        if isinstance(var.mutable_local, (AttributeMutationExisting, AttributeMutationNew)) and isinstance(var, variables.NewCellVariable):
            cg.load_import_from(utils.__name__, 'make_cell')
            cg.extend_output(create_call_function(0, True))
            cg.add_cache(var)
            if isinstance(var.mutable_local, AttributeMutationNew):
                var.mutable_local.source = LocalSource(cg.tempvars[var])
        elif isinstance(var.mutable_local, AttributeMutationNew):
            if isinstance(var, variables.AutogradFunctionContextVariable):
                unimplemented('AutogradFunctionContextVariable escaped')
            if '__call_nn_module_init' in self.store_attr_mutations.get(var.mutable_local, {}):
                assert isinstance(var, variables.UnspecializedNNModuleVariable)
                cg.load_import_from(utils.__name__, 'nn_module_new')
            else:
                cg.load_import_from(utils.__name__, 'object_new')
            cg(var.mutable_local.cls_source)
            cg.extend_output(create_call_function(1, True))
            cg.add_cache(var)
            var.mutable_local.source = LocalSource(cg.tempvars[var])
        elif var in cg.tempvars:
            assert cg.tempvars.get(var) is None
            cg(var.mutable_local.source)
            cg.add_cache(var)
    for ctx, args in self.save_for_backward:
        cg(ctx.source)
        cg.extend_output([create_instruction('LOAD_METHOD', argval='save_for_backward')])
        for arg in args:
            cg(arg)
        cg.extend_output([*create_call_method(len(args)), create_instruction('POP_TOP')])