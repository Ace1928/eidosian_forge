import time
from unittest import mock
import uuid
from pycadf import attachment
from pycadf import cadftype
from pycadf import credential
from pycadf import endpoint
from pycadf import event
from pycadf import geolocation
from pycadf import host
from pycadf import identifier
from pycadf import measurement
from pycadf import metric
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import tag
from pycadf.tests import base
from pycadf import timestamp
def test_measurement(self):
    measure_val = measurement.Measurement(result='100', metric=metric.Metric(), metricId=identifier.generate_uuid(), calculatedBy=resource.Resource(typeURI='storage'))
    self.assertEqual(False, measure_val.is_valid())
    dict_measure_val = measure_val.as_dict()
    for key in measurement.MEASUREMENT_KEYNAMES:
        self.assertIn(key, dict_measure_val)
    measure_val = measurement.Measurement(result='100', metric=metric.Metric(), calculatedBy=resource.Resource(typeURI='storage'))
    self.assertEqual(True, measure_val.is_valid())
    measure_val = measurement.Measurement(result='100', metricId=identifier.generate_uuid(), calculatedBy=resource.Resource(typeURI='storage'))
    self.assertEqual(True, measure_val.is_valid())