import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def aws_partition(self, value):
    """Match a region string to an AWS partition.

        :type value: str
        :rtype: dict
        """
    partitions = self.partitions_data['partitions']
    if value is not None:
        for partition in partitions:
            if self.is_partition_match(value, partition):
                return self.format_partition_output(partition)
    aws_partition = partitions[0]
    return self.format_partition_output(aws_partition)