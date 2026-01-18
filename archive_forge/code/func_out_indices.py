import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
@out_indices.setter
def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
    """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
    self._out_features, self._out_indices = get_aligned_output_features_output_indices(out_features=None, out_indices=out_indices, stage_names=self.stage_names)