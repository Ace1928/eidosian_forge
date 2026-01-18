from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.rds.parametergroup import ParameterGroup
from boto.rds.statusinfo import StatusInfo
from boto.rds.dbsubnetgroup import DBSubnetGroup
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.resultset import ResultSet
@property
def security_group(self):
    """
        Provide backward compatibility for previous security_group
        attribute.
        """
    if len(self.security_groups) > 0:
        return self.security_groups[-1]
    else:
        return None