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
def get_best_trials(self, num_trials=1):
    """Returns the best `Trial`s."""
    trials = [t for t in self.trials.values() if t.status == trial_module.TrialStatus.COMPLETED]
    sorted_trials = sorted(trials, key=lambda trial: trial.score, reverse=self.objective.direction == 'max')
    if len(sorted_trials) < num_trials:
        sorted_trials = sorted_trials + [t for t in self.trials.values() if t.status != trial_module.TrialStatus.COMPLETED]
    return sorted_trials[:num_trials]