import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def on_trial_end(self, trial):
    if self.verbose < 1:
        return
    utils.try_clear()
    time_taken_str = self.format_duration(datetime.now() - self.trial_start[trial.trial_id])
    print(f'Trial {self.trial_number[trial.trial_id]} Complete [{time_taken_str}]')
    if trial.score is not None:
        print(f'{self.oracle.objective.name}: {trial.score}')
    print()
    best_trials = self.oracle.get_best_trials()
    best_score = best_trials[0].score if len(best_trials) > 0 else None
    print(f'Best {self.oracle.objective.name} So Far: {best_score}')
    time_elapsed_str = self.format_duration(datetime.now() - self.search_start)
    print(f'Total elapsed time: {time_elapsed_str}')