import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path
def function_to_taskspec(func: Callable, is_config: Callable[[List[Dict[str, Any]]], List[bool]], deterministic: bool=True, lazy: bool=False) -> TaskSpec:
    specs = inspect.getfullargspec(func)
    sig = inspect.signature(func)
    annotations = get_type_hints(func)
    assert_or_throw(specs.varargs is None and specs.varkw is None and (len(specs.kwonlyargs) == 0), "Function can't have varargs or kwargs")
    inputs: List[InputSpec] = []
    configs: List[ConfigSpec] = []
    outputs: List[OutputSpec] = []
    arr: List[Dict[str, Any]] = []
    for k, w in sig.parameters.items():
        anno = annotations.get(k, w.annotation)
        a = _parse_annotation(anno)
        a['name'] = k
        if w.default == inspect.Parameter.empty:
            a['required'] = True
        else:
            a['required'] = False
            a['default_value'] = w.default
        arr.append(a)
    cfg = is_config(arr)
    for i in range(len(cfg)):
        if cfg[i]:
            configs.append(ConfigSpec(**arr[i]))
        else:
            assert_or_throw(arr[i]['required'], f'{arr[i]}: dependency must not have default value')
            inputs.append(InputSpec(**arr[i]))
    n = 0
    anno = annotations.get('return', sig.return_annotation)
    is_multiple = _is_tuple(anno)
    for x in list(anno.__args__) if is_multiple else [anno]:
        if x == inspect.Parameter.empty or x is type(None):
            continue
        a = _parse_annotation(x)
        a['name'] = f'_{n}'
        outputs.append(OutputSpec(**a))
        n += 1
    metadata = dict(__interfaceless_func=get_full_type_path(func))
    return TaskSpec(configs, inputs, outputs, _interfaceless_wrapper, metadata, deterministic=deterministic, lazy=lazy)