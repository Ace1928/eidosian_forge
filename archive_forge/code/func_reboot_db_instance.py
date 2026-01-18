import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def reboot_db_instance(self, db_instance_identifier, force_failover=None):
    """
        Rebooting a DB instance restarts the database engine service.
        A reboot also applies to the DB instance any modifications to
        the associated DB parameter group that were pending. Rebooting
        a DB instance results in a momentary outage of the instance,
        during which the DB instance status is set to rebooting. If
        the RDS instance is configured for MultiAZ, it is possible
        that the reboot will be conducted through a failover. An
        Amazon RDS event is created when the reboot is completed.

        If your DB instance is deployed in multiple Availability
        Zones, you can force a failover from one AZ to the other
        during the reboot. You might force a failover to test the
        availability of your DB instance deployment or to restore
        operations to the original AZ after a failover occurs.

        The time required to reboot is a function of the specific
        database engine's crash recovery process. To improve the
        reboot time, we recommend that you reduce database activities
        as much as possible during the reboot process to reduce
        rollback activity for in-transit transactions.

        :type db_instance_identifier: string
        :param db_instance_identifier:
        The DB instance identifier. This parameter is stored as a lowercase
            string.

        Constraints:


        + Must contain from 1 to 63 alphanumeric characters or hyphens
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type force_failover: boolean
        :param force_failover: When `True`, the reboot will be conducted
            through a MultiAZ failover.
        Constraint: You cannot specify `True` if the instance is not configured
            for MultiAZ.

        """
    params = {'DBInstanceIdentifier': db_instance_identifier}
    if force_failover is not None:
        params['ForceFailover'] = str(force_failover).lower()
    return self._make_request(action='RebootDBInstance', verb='POST', path='/', params=params)