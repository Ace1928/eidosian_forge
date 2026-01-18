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
def deserialize_xgb_model(model: str, xgb_model_creator: Callable[[], XGBModel]) -> XGBModel:
    """
    Deserialize an xgboost.XGBModel instance from the input model.
    """
    xgb_model = xgb_model_creator()
    xgb_model.load_model(bytearray(model.encode('utf-8')))
    return xgb_model