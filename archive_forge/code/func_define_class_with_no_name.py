import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
def define_class_with_no_name():

    class FakeTraceWithMetaclassNoName(FakeTracedCls, metaclass=profiler.TracedMeta):
        pass