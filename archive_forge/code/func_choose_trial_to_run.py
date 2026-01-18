import logging
from typing import Dict, Optional, TYPE_CHECKING
from ray.tune.schedulers.trial_scheduler import TrialScheduler
from ray.tune.schedulers.hyperband import HyperBandScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def choose_trial_to_run(self, tune_controller: 'TuneController', allow_recurse: bool=True) -> Optional[Trial]:
    """Fair scheduling within iteration by completion percentage.

        List of trials not used since all trials are tracked as state
        of scheduler. If iteration is occupied (ie, no trials to run),
        then look into next iteration.
        """
    for hyperband in self._hyperbands:
        scrubbed = [b for b in hyperband if b is not None]
        for bracket in scrubbed:
            for trial in bracket.current_trials():
                if trial.status == Trial.PAUSED and trial in bracket.trials_to_unpause or trial.status == Trial.PENDING:
                    return trial
    if not any((t.status == Trial.RUNNING for t in tune_controller.get_trials())):
        for hyperband in self._hyperbands:
            for bracket in hyperband:
                if bracket and any((trial.status == Trial.PAUSED for trial in bracket.current_trials())):
                    logger.debug('Processing bracket since no trial is running.')
                    self._process_bracket(tune_controller, bracket)
                    if allow_recurse and any((trial.status == Trial.PAUSED and trial in bracket.trials_to_unpause or trial.status == Trial.PENDING for trial in bracket.current_trials())):
                        return self.choose_trial_to_run(tune_controller, allow_recurse=False)
    return None