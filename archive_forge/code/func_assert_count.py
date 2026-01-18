import json
from fugue import (
import cloudpickle
import fugue
from tune._utils.serialization import from_base64
from tune.constants import (
from tune.concepts.dataset import TuneDatasetBuilder, _to_trail_row, TuneDataset
from tune.concepts.space import Grid, Rand
from tune.concepts.space.spaces import Space
from tune.concepts.flow import Trial
def assert_count(df: DataFrame, n: int, schema=None) -> None:
    assert len(df.as_array()) == n
    if schema is not None:
        assert df.schema == schema