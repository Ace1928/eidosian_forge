import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow import task
def subflow_factory(prefix):

    def pr(what):
        return '%s-%s' % (prefix, what)
    return lf.Flow(pr('flow')).add(FetchNumberTask(pr('fetch'), provides=pr('number'), rebind=[pr('person')]), CallTask(pr('call'), rebind=[pr('person'), pr('number')]))