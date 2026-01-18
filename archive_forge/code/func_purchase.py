from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
def purchase(self, instance_count=1, dry_run=False):
    return self.connection.purchase_reserved_instance_offering(self.id, instance_count, dry_run=dry_run)