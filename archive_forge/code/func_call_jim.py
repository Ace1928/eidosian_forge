import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.types import notifier
def call_jim(context):
    print('Calling jim.')
    print('Context = %s' % sorted(context.items(), key=lambda x: x[0]))