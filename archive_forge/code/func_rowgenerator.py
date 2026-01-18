from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def rowgenerator(rec):
    transmf = {'male': 'M', 'female': 'F'}
    yield [rec['id'], 'gender', transmf[rec['sex']] if rec['sex'] in transmf else rec['sex']]
    yield [rec['id'], 'age_months', rec['age'] * 12]
    yield [rec['id'], 'bmi', rec['weight'] / rec['height'] ** 2]