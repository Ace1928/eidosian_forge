import logging
from typing import Dict, Optional, TYPE_CHECKING
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def on_trial_add(self, tune_controller: 'TuneController', trial: Trial):
    """Adds new trial.

        On a new trial add, if current bracket is not filled, add to current
        bracket. Else, if current band is not filled, create new bracket, add
        to current bracket. Else, create new iteration, create new bracket,
        add to bracket.
        """
    if not self._metric or not self._metric_op:
        raise ValueError('{} has been instantiated without a valid `metric` ({}) or `mode` ({}) parameter. Either pass these parameters when instantiating the scheduler, or pass them as parameters to `tune.TuneConfig()`'.format(self.__class__.__name__, self._metric, self._mode))
    cur_bracket = self._state['bracket']
    cur_band = self._hyperbands[self._state['band_idx']]
    if cur_bracket is None or cur_bracket.filled():
        retry = True
        while retry:
            if self._cur_band_filled():
                cur_band = []
                self._hyperbands.append(cur_band)
                self._state['band_idx'] += 1
            s = self._s_max_1 - len(cur_band) - 1
            assert s >= 0, 'Current band is filled!'
            if self._get_r0(s) == 0:
                logger.debug('BOHB: Bracket too small - Retrying...')
                cur_bracket = None
            else:
                retry = False
                cur_bracket = self._create_bracket(s)
            cur_band.append(cur_bracket)
            self._state['bracket'] = cur_bracket
    self._state['bracket'].add_trial(trial)
    self._trial_info[trial] = (cur_bracket, self._state['band_idx'])