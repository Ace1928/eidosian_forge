from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.rds.parametergroup import ParameterGroup
from boto.rds.statusinfo import StatusInfo
from boto.rds.dbsubnetgroup import DBSubnetGroup
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.resultset import ResultSet
@property
def parameter_group(self):
    """
        Provide backward compatibility for previous parameter_group
        attribute.
        """
    if len(self.parameter_groups) > 0:
        return self.parameter_groups[-1]
    else:
        return None