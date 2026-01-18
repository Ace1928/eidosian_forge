from typing import Dict, Iterable, List, Tuple
import torch
def del_tensor(self, name: str) -> None:
    """
        Delete the attribute specified by the given path.

        For example, to delete the attribute mod.layer1.conv1.weight,
        use accessor.del_tensor("layer1.conv1.weight")
        """
    prefix, _, attr = name.rpartition('.')
    submodule = self.get_submodule(prefix)
    try:
        delattr(submodule, attr)
    except AttributeError as ex:
        raise AttributeError(f'{submodule._get_name()} has no attribute `{name}`') from ex