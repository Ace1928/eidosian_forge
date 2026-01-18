import copy
import glob
import itertools
import os
import uuid
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.air._internal.usage import tag_searcher
from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI
def create_trial(self, resolved_vars, spec):
    trial_id = self.uuid_prefix + '%05d' % self.counter
    experiment_tag = str(self.counter)
    if resolved_vars:
        experiment_tag += '_{}'.format(format_vars(resolved_vars))
    self.counter += 1
    return _create_trial_from_spec(spec, self.parser, evaluated_params=_flatten_resolved_vars(resolved_vars), trial_id=trial_id, experiment_tag=experiment_tag)