from typing import Any, Callable, Dict, Hashable, Optional, Sequence, Tuple, Union, cast
from ..config import registry
from ..model import Model
from ..types import DTypes, Ints1d, Ints2d
from ..util import is_xp_array, to_numpy
@registry.layers('remap_ids.v2')
def remap_ids_v2(mapping_table: Optional[Union[Dict[int, int], Dict[str, int]]]=None, default: int=0, *, column: Optional[int]=None) -> Model[InT, OutT]:
    """Remap string or integer inputs using a mapping table,
    usually as a preprocessing step before embeddings.
    The mapping table can be passed in on input,
    or updated after the layer has been created.
    The mapping table is stored in the "mapping_table" attribute.
    Two dimensional arrays can be provided as input in which case
    the 'column' chooses which column to process. This is useful
    to work together with FeatureExtractor in spaCy.
    """
    return Model('remap_ids', forward_v2, attrs={'mapping_table': mapping_table, 'default': default, 'column': column})