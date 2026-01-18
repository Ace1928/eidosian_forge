import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
from . import callback
from .basic import (Booster, Dataset, LightGBMError, _choose_param_value, _ConfigAliases, _InnerPredictor,
from .compat import SKLEARN_INSTALLED, _LGBMBaseCrossValidator, _LGBMGroupKFold, _LGBMStratifiedKFold
def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
    """Call methods with each booster, and concatenate their results."""
    ret = []
    for booster in self.boosters:
        ret.append(getattr(booster, name)(*args, **kwargs))
    return ret