from tests.integration import ServiceCertVerificationTest
from tests.compat import unittest
import boto.ec2.elb
def sample_service_call(self, conn):
    conn.get_all_load_balancers()