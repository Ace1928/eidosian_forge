from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
from ray.actor import ActorHandle
from ray.data import Dataset
def split_dataset(dataset_or_pipeline):
    return dataset_or_pipeline.split(len(training_worker_handles), equal=True, locality_hints=training_worker_handles)