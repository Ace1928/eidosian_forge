import inspect
import logging
import os
import sys
import uuid
from threading import Thread
from typing import Any, Callable, Dict, Optional, Set, Type
import pyspark
from pyspark import BarrierTaskContext, SparkContext, SparkFiles, TaskContext
from pyspark.sql.session import SparkSession
from xgboost import Booster, XGBModel, collective
from xgboost.tracker import RabitTracker
def deserialize_booster(model: str) -> Booster:
    """
    Deserialize an xgboost.core.Booster from the input ser_model_string.
    """
    booster = Booster()
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f'{uuid.uuid4()}.json')
    with open(tmp_file_name, 'w', encoding='utf-8') as f:
        f.write(model)
    booster.load_model(tmp_file_name)
    return booster