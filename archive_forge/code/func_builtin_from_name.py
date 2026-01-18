from jedi.inference.compiled.value import CompiledValue, CompiledName, \
from jedi.inference.base_value import LazyValueWrapper
def builtin_from_name(inference_state, string):
    typing_builtins_module = inference_state.builtins_module
    if string in ('None', 'True', 'False'):
        builtins, = typing_builtins_module.non_stub_value_set
        filter_ = next(builtins.get_filters())
    else:
        filter_ = next(typing_builtins_module.get_filters())
    name, = filter_.get(string)
    value, = name.infer()
    return value