import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
def replace_all_uses(self, old: str, new: str):
    """
        Replace all uses of the old name with new name in the signature.
        """
    assert isinstance(old, str)
    assert isinstance(new, str)
    for o in self.output_specs:
        if isinstance(o.arg, TensorArgument):
            if o.arg.name == old:
                o.arg.name = new