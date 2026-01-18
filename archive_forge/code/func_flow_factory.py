import os
from taskflow.patterns import linear_flow as lf
from taskflow import task
def flow_factory():
    return lf.Flow('example').add(TestTask(name='first'), UnfortunateTask(name='boom'), TestTask(name='second'))