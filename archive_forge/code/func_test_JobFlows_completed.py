import xml.sax
from boto import handler
from boto.emr import emrobject
from boto.resultset import ResultSet
from tests.compat import unittest
def test_JobFlows_completed(self):
    [jobflow] = self._parse_xml(JOB_FLOW_COMPLETED, [('member', emrobject.JobFlow)])
    self._assert_fields(jobflow, creationdatetime='2010-10-21T01:00:25Z', startdatetime='2010-10-21T01:03:59Z', enddatetime='2010-10-21T01:44:18Z', state='COMPLETED', instancecount='10', jobflowid='j-3H3Q13JPFLU22', loguri='s3n://example.emrtest.scripts/jobflow_logs/', name='RealJobFlowName', availabilityzone='us-east-1b', slaveinstancetype='m1.large', masterinstancetype='m1.large', ec2keyname='myubersecurekey', keepjobflowalivewhennosteps='false')
    self.assertEquals(6, len(jobflow.steps))
    self.assertEquals(2, len(jobflow.instancegroups))