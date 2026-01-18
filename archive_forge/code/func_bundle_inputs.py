from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def bundle_inputs(model: torch.jit.ScriptModule, inputs: Union[Optional[Sequence[Tuple[Any, ...]]], Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]]], info: Optional[Union[List[str], Dict[Callable, List[str]]]]=None, *, _receive_inflate_expr: Optional[List[str]]=None) -> torch.jit.ScriptModule:
    """Create and return a copy of the specified model with inputs attached.

    The original model is not mutated or changed in any way.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    If inputs is passed in as a list then the inputs will be bundled for 'forward'.
    If inputs is instead passed in as a map then all the methods specified in the map
    will have their corresponding inputs bundled. Info should match watchever type is
    chosen for the inputs.

    The returned model will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions will also be defined on the returned module:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs
        can be provided instead.

        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and
    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception('Only ScriptModule is supported.')
    ignored_methods, ignored_attrs = _get_bundled_inputs_attributes_and_methods(model)
    clone = torch._C._hack_do_not_use_clone_module_with_class(model._c, ignored_methods, ignored_attrs)
    cloned_module = wrap_cpp_module(clone)
    if isinstance(inputs, dict):
        assert isinstance(info, dict) or info is None
        augment_many_model_functions_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    else:
        assert isinstance(info, list) or info is None
        augment_model_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    return cloned_module