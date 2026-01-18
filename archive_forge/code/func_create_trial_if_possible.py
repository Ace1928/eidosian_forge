import copy
import logging
from typing import Dict, List, Optional, Union
from ray.tune.error import TuneError
from ray.tune.experiment import Experiment, _convert_to_experiment_list
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.search import Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.tune.search.variant_generator import format_vars, _resolve_nested_dict
from ray.tune.experiment import Trial
from ray.tune.utils.util import (
from ray.util.annotations import DeveloperAPI
def create_trial_if_possible(self, experiment_spec: Dict) -> Optional[Trial]:
    logger.debug('creating trial')
    trial_id = Trial.generate_id()
    suggested_config = self.searcher.suggest(trial_id)
    if suggested_config == Searcher.FINISHED:
        self._finished = True
        logger.debug('Searcher has finished.')
        return
    if suggested_config is None:
        return
    spec = copy.deepcopy(experiment_spec)
    spec['config'] = merge_dicts(spec['config'], copy.deepcopy(suggested_config))
    flattened_config = _resolve_nested_dict(spec['config'])
    self._counter += 1
    tag = '{0}_{1}'.format(str(self._counter), format_vars(flattened_config))
    trial = _create_trial_from_spec(spec, self._parser, evaluated_params=flatten_dict(suggested_config), experiment_tag=tag, trial_id=trial_id)
    return trial