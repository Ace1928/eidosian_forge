import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def get_aligned_output_features_output_indices(out_features: Optional[List[str]], out_indices: Optional[Union[List[int], Tuple[int]]], stage_names: List[str]) -> Tuple[List[str], List[int]]:
    """
    Get the `out_features` and `out_indices` so that they are aligned.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: they are verified to be aligned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    output_features, output_indices = _align_output_features_output_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    return (output_features, output_indices)