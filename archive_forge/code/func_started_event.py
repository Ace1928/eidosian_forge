import warnings
import numpy
from sacred.dependencies import get_digest
from sacred.observers import RunObserver
import wandb
def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
    self.__update_config(config)