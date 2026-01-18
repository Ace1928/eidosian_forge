import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
import wandb
from wandb.util import get_module
def get_updated_kwargs(pipeline: Any, args: Sequence[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_call_parameters = list(inspect.signature(pipeline.__call__).parameters.items())
    for idx, arg in enumerate(args):
        kwargs[pipeline_call_parameters[idx][0]] = arg
    for pipeline_parameter in pipeline_call_parameters:
        if pipeline_parameter[0] not in kwargs:
            kwargs[pipeline_parameter[0]] = pipeline_parameter[1].default
    if 'generator' in kwargs:
        generator = kwargs['generator']
        kwargs['generator'] = {'seed': generator.initial_seed(), 'device': generator.device, 'random_state': generator.get_state().cpu().numpy().tolist()} if generator is not None else None
    if 'ip_adapter_image' in kwargs:
        if kwargs['ip_adapter_image'] is not None:
            wandb.log({'IP-Adapter-Image': wandb.Image(kwargs['ip_adapter_image'])})
    return kwargs