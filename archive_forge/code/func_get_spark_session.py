import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def get_spark_session():
    from pyspark.sql import SparkSession
    spark_session = SparkSession.getActiveSession()
    if spark_session is None:
        raise RuntimeError("Spark session haven't been initiated yet. Please use `SparkSession.builder` to create a spark session and connect to a spark cluster.")
    return spark_session