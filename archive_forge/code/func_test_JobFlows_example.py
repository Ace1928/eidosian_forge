import xml.sax
from boto import handler
from boto.emr import emrobject
from boto.resultset import ResultSet
from tests.compat import unittest
def test_JobFlows_example(self):
    [jobflow] = self._parse_xml(JOB_FLOW_EXAMPLE, [('member', emrobject.JobFlow)])
    self._assert_fields(jobflow, creationdatetime='2009-01-28T21:49:16Z', startdatetime='2009-01-28T21:49:16Z', state='STARTING', instancecount='4', jobflowid='j-3UN6WX5RRO2AG', loguri='mybucket/subdir/', name='MyJobFlowName', availabilityzone='us-east-1a', slaveinstancetype='m1.small', masterinstancetype='m1.small', ec2keyname='myec2keyname', keepjobflowalivewhennosteps='true')