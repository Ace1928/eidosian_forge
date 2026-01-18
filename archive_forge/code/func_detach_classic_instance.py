from boto.ec2.ec2object import TaggedEC2Object
def detach_classic_instance(self, instance_id, dry_run=False):
    """
        Unlinks a linked EC2-Classic instance from a VPC. After the instance
        has been unlinked, the VPC security groups are no longer associated
        with it. An instance is automatically unlinked from a VPC when
        it's stopped.

        :type intance_id: str
        :param instance_is: The ID of the VPC to which the instance is linked.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    return self.connection.detach_classic_link_vpc(vpc_id=self.id, instance_id=instance_id, dry_run=dry_run)