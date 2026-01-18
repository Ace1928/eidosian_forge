import collections
import copy
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
@oracle_module.synchronized
def end_trial(self, trial):
    super().end_trial(trial)
    self._populate_next.append(trial.trial_id)