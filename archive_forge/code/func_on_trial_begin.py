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
def on_trial_begin(self, trial):
    if self.verbose < 1:
        return
    start_time = datetime.now()
    self.trial_start[trial.trial_id] = start_time
    if self.search_start is None:
        self.search_start = start_time
    current_number = len(self.oracle.trials)
    self.trial_number[trial.trial_id] = current_number
    print()
    print(f'Search: Running Trial #{current_number}')
    print()
    self.show_hyperparameter_table(trial)
    print()