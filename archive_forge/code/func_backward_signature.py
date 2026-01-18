import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def backward_signature(self) -> Optional[ExportBackwardSignature]:
    loss_output = None
    gradients_to_parameters: Dict[str, str] = {}
    gradients_to_user_inputs: Dict[str, str] = {}
    for spec in self.output_specs:
        if spec.kind == OutputKind.LOSS_OUTPUT:
            assert loss_output is None
            assert isinstance(spec.arg, TensorArgument)
            loss_output = spec.arg.name
        elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
            assert isinstance(spec.target, str)
            assert isinstance(spec.arg, TensorArgument)
            gradients_to_parameters[spec.arg.name] = spec.target
        elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
            assert isinstance(spec.target, str)
            assert isinstance(spec.arg, TensorArgument)
            gradients_to_user_inputs[spec.arg.name] = spec.target
    if loss_output is None:
        return None
    return ExportBackwardSignature(loss_output=loss_output, gradients_to_parameters=gradients_to_parameters, gradients_to_user_inputs=gradients_to_user_inputs)