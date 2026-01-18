from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
def basic_generator():
    return base_model.ReportModel(data={'string': 'value', 'int': 1})