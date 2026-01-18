from adagio.shells.interfaceless import function_to_taskspec, _get_origin_type, _parse_annotation
from typing import Any, Dict, List, Union, Optional, Tuple
from pytest import raises
import inspect
def test__get_origin_type():
    assert _get_origin_type(Any) is object
    assert _get_origin_type(Dict[str, Any]) is dict
    assert _get_origin_type(List[str]) is list
    assert _get_origin_type(List[Any]) is list
    assert _get_origin_type(Tuple[int, str]) is tuple
    assert _get_origin_type(Union[int, str], False) is Union
    assert _get_origin_type(Union[None]) is type(None)
    assert _get_origin_type(int) is int