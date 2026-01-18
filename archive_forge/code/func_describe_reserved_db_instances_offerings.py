import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def describe_reserved_db_instances_offerings(self, reserved_db_instances_offering_id=None, db_instance_class=None, duration=None, product_description=None, offering_type=None, multi_az=None, max_records=None, marker=None):
    """
        Lists available reserved DB instance offerings.

        :type reserved_db_instances_offering_id: string
        :param reserved_db_instances_offering_id: The offering identifier
            filter value. Specify this parameter to show only the available
            offering that matches the specified reservation identifier.
        Example: `438012d3-4052-4cc7-b2e3-8d3372e0e706`

        :type db_instance_class: string
        :param db_instance_class: The DB instance class filter value. Specify
            this parameter to show only the available offerings matching the
            specified DB instance class.

        :type duration: string
        :param duration: Duration filter value, specified in years or seconds.
            Specify this parameter to show only reservations for this duration.
        Valid Values: `1 | 3 | 31536000 | 94608000`

        :type product_description: string
        :param product_description: Product description filter value. Specify
            this parameter to show only the available offerings matching the
            specified product description.

        :type offering_type: string
        :param offering_type: The offering type filter value. Specify this
            parameter to show only the available offerings matching the
            specified offering type.
        Valid Values: `"Light Utilization" | "Medium Utilization" | "Heavy
            Utilization" `

        :type multi_az: boolean
        :param multi_az: The Multi-AZ filter value. Specify this parameter to
            show only the available offerings matching the specified Multi-AZ
            parameter.

        :type max_records: integer
        :param max_records: The maximum number of records to include in the
            response. If more than the `MaxRecords` value is available, a
            pagination token called a marker is included in the response so
            that the following results can be retrieved.
        Default: 100

        Constraints: minimum 20, maximum 100

        :type marker: string
        :param marker: An optional pagination token provided by a previous
            request. If this parameter is specified, the response includes only
            records beyond the marker, up to the value specified by
            `MaxRecords`.

        """
    params = {}
    if reserved_db_instances_offering_id is not None:
        params['ReservedDBInstancesOfferingId'] = reserved_db_instances_offering_id
    if db_instance_class is not None:
        params['DBInstanceClass'] = db_instance_class
    if duration is not None:
        params['Duration'] = duration
    if product_description is not None:
        params['ProductDescription'] = product_description
    if offering_type is not None:
        params['OfferingType'] = offering_type
    if multi_az is not None:
        params['MultiAZ'] = str(multi_az).lower()
    if max_records is not None:
        params['MaxRecords'] = max_records
    if marker is not None:
        params['Marker'] = marker
    return self._make_request(action='DescribeReservedDBInstancesOfferings', verb='POST', path='/', params=params)