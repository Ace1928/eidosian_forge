from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_fieldmap_record_access():
    table = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, 'female', 17, 1.78, 74.4), (4, 'male', 21, 1.33, 45.2), (5, '-', 25, 1.65, 51.9))
    mappings = OrderedDict()
    mappings['subject_id'] = 'id'
    mappings['gender'] = ('sex', {'male': 'M', 'female': 'F'})
    mappings['age_months'] = ('age', lambda v: v * 12)
    mappings['bmi'] = lambda rec: rec.weight / rec.height ** 2
    actual = fieldmap(table, mappings)
    expect = (('subject_id', 'gender', 'age_months', 'bmi'), (1, 'M', 16 * 12, 62.0 / 1.45 ** 2), (2, 'F', 19 * 12, 55.4 / 1.34 ** 2), (3, 'F', 17 * 12, 74.4 / 1.78 ** 2), (4, 'M', 21 * 12, 45.2 / 1.33 ** 2), (5, '-', 25 * 12, 51.9 / 1.65 ** 2))
    ieq(expect, actual)
    ieq(expect, actual)