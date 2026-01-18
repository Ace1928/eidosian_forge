import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def dirtyResource(test):
    make_counter.dirtied(test._default)