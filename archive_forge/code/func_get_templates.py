from typing import Any, Iterable
import numpy as np
from triad import assert_or_throw
from tune._utils import product
from tune.concepts.space.parameters import TuningParametersTemplate
def get_templates() -> Iterable[TuningParametersTemplate]:
    o = other if isinstance(other, Space) else Space(other)
    yield from self
    yield from o