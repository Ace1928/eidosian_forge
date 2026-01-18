from typing import Dict, Union
from torchgen.model import NativeFunctionsGroup, NativeFunctionsViewGroup
def func_name_base_str(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> str:
    if isinstance(g, NativeFunctionsGroup):
        return str(g.functional.func.name.name.base)
    else:
        return str(g.view.root_name)