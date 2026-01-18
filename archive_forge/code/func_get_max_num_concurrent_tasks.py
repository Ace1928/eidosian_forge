import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def get_max_num_concurrent_tasks(spark_context, resource_profile):
    """Gets the current max number of concurrent tasks."""
    ssc = spark_context._jsc.sc()
    if resource_profile is not None:

        def dummpy_mapper(_):
            pass
        spark_context.parallelize([1], 1).withResources(resource_profile).map(dummpy_mapper).collect()
        return ssc.maxNumConcurrentTasks(resource_profile._java_resource_profile)
    else:
        return ssc.maxNumConcurrentTasks(ssc.resourceProfileManager().defaultResourceProfile())